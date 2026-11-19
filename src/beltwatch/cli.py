"""Train, calibrate, evaluate and write an offline replay."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import save_file
from torch import Tensor

from beltwatch.data import CONTEXT, FAULTS, Clip, make_split
from beltwatch.metrics import MethodMetrics, calibrated_threshold, evaluate
from beltwatch.model import SIGREG_WEIGHT, BeltModel, fit_model, frame_scores, latent_spread
from beltwatch.report import write_report


@dataclass(frozen=True)
class Profile:
    train_scenes: int
    calibration_scenes: int
    test_scenes: int
    epochs: int


FULL = Profile(160, 72, 80, 14)
QUICK = Profile(12, 12, 4, 2)
ALPHA = 0.10
_CALIBRATION_SEED_BIT = 1 << 30
_TEST_SEED_BIT = 1 << 31


def _frames(clips: Sequence[Clip]) -> Tensor:
    return torch.from_numpy(np.stack([clip.frames for clip in clips])).float().div_(255.0)


def _pixel_scores(clips: Tensor, *, extrapolate: bool) -> Tensor:
    previous = clips[:, CONTEXT - 1 : -1]
    observed = clips[:, CONTEXT:]
    predicted = previous
    if extrapolate:
        predicted = (2 * previous - clips[:, CONTEXT - 2 : -2]).clamp(0, 1)
    return (observed - predicted).square().mean(dim=(2, 3))


def _shift_scores(clips: Tensor) -> Tensor:
    """Fit a fixed horizontal translation from each clip's normal warm-up."""
    radius = 3
    shifts = range(-radius, radius + 1)
    width = clips.shape[-1]
    middle = slice(radius, width - radius)
    warm_target = clips[:, 1:CONTEXT, :, middle]
    initial = clips[:, : CONTEXT - 1]
    errors = torch.stack(
        [(warm_target - initial[..., radius - s : width - radius - s]).square().mean((1, 2, 3)) for s in shifts],
        dim=0,
    )
    chosen = errors.argmin(dim=0)
    previous = clips[:, CONTEXT - 1 : -1]
    observed = clips[:, CONTEXT:, :, middle]
    scores = torch.stack(
        [(observed - previous[..., radius - s : width - radius - s]).square().mean((2, 3)) for s in shifts],
        dim=0,
    )
    return scores[chosen, torch.arange(len(clips))]


@torch.inference_mode()
def _batched(score: Callable[[Tensor], Tensor], frames: Tensor) -> NDArray[np.float64]:
    return torch.cat([score(chunk) for chunk in frames.split(8)]).numpy().astype(np.float64)


def _run(out: Path, *, profile: Profile, seed: int, threads: int) -> None:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    train = _frames(make_split("train", profile.train_scenes, seed=seed))
    calibration_seed = seed ^ _CALIBRATION_SEED_BIT
    test_seed = seed ^ _TEST_SEED_BIT
    calibration = _frames(make_split("calibration", profile.calibration_scenes, seed=calibration_seed))
    test = make_split("test", profile.test_scenes, seed=test_seed)
    test_frames = _frames(test)

    methods: dict[str, MethodMetrics] = {}
    training: dict[str, object] = {}
    latent_scores: NDArray[np.float64] | None = None
    primary_model: BeltModel | None = None

    for name, weight in (("latent_sigreg", SIGREG_WEIGHT), ("latent_no_sigreg", 0.0)):
        print(f"Training {name} ({profile.epochs} epochs on CPU)...", file=sys.stderr, flush=True)
        model, history = fit_model(train, epochs=profile.epochs, seed=seed, sigreg_weight=weight)

        score = partial(frame_scores, model)
        threshold = calibrated_threshold(_batched(score, calibration), alpha=ALPHA)
        held_out = _batched(score, test_frames)
        methods[name] = evaluate(test, held_out, threshold=threshold)
        training[name] = {"epochs": history, "calibration_latent_spread": latent_spread(model, calibration)}
        if name == "latent_sigreg":
            latent_scores, primary_model = held_out, model

    for name, extrapolate in (("pixel_delta", False), ("pixel_linear", True)):
        threshold = calibrated_threshold(_pixel_scores(calibration, extrapolate=extrapolate).numpy(), alpha=ALPHA)
        methods[name] = evaluate(test, _pixel_scores(test_frames, extrapolate=extrapolate).numpy(), threshold=threshold)

    threshold = calibrated_threshold(_batched(_shift_scores, calibration), alpha=ALPHA)
    methods["pixel_shift"] = evaluate(test, _batched(_shift_scores, test_frames), threshold=threshold)

    assert primary_model is not None and latent_scores is not None
    metrics = {
        "schema_version": 1,
        "config": {
            **asdict(profile),
            "seed": seed,
            "calibration_seed": calibration_seed,
            "test_seed": test_seed,
            "threads": threads,
            "context_frames": CONTEXT,
            "target_false_alarm": ALPHA,
            "sigreg_weight": SIGREG_WEIGHT,
            "parameters": sum(parameter.numel() for parameter in primary_model.parameters()),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        "training": training,
        "methods": methods,
    }
    demo_indices = [
        next(index for index, clip in enumerate(test) if clip.fault == fault) for fault in ("normal", *FAULTS)
    ]
    write_report(out, [(test[i], latent_scores[i]) for i in demo_indices], metrics)
    save_file(
        {name: tensor.contiguous() for name, tensor in primary_model.state_dict().items()}, out / "model.safetensors"
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Artifacts: {out.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and return a process exit status."""
    parser = argparse.ArgumentParser(prog="beltwatch", description=__doc__)
    run = parser.add_subparsers(dest="command", required=True).add_parser("run", help="run the CPU benchmark")
    run.add_argument("--out", type=Path, required=True, help="new directory for the run artifacts")
    run.add_argument("--quick", action="store_true", help="exercise the pipeline with a tiny split; not for comparison")
    run.add_argument("--seed", type=int, default=2026, help="seed shared by the simulator and training")
    run.add_argument("--threads", type=int, choices=range(1, 9), default=2, metavar="1..8")
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be in 0..4294967295")
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Output path already exists: {args.out}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Cannot create output directory: {exc}", file=sys.stderr)
        return 2
    try:
        _run(args.out, profile=QUICK if args.quick else FULL, seed=args.seed, threads=args.threads)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Run failed: {exc}. A partial run may remain in {args.out}.", file=sys.stderr)
        return 1
    return 0
