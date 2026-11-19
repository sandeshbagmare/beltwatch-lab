import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from beltwatch.cli import _shift_scores, main


def test_cli_rejects_existing_output_before_writing(tmp_path: Path, capsys):
    existing = tmp_path / "existing"
    existing.mkdir()
    original = existing / "keep.txt"
    original.write_text("user file", encoding="utf-8")
    assert main(["run", "--quick", "--out", str(existing)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert list(existing.iterdir()) == [original]
    assert original.read_text(encoding="utf-8") == "user file"


def test_cli_benchmarks_and_generates_offline_artifacts(tmp_path: Path):
    out = tmp_path / "generated"
    assert main(["run", "--quick", "--out", str(out), "--seed", "42"]) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))

    assert set(metrics["methods"]) == {
        "latent_sigreg",
        "latent_no_sigreg",
        "pixel_delta",
        "pixel_linear",
        "pixel_shift",
    }
    assert metrics["config"]["seed"] == 42
    assert len({42, metrics["config"]["calibration_seed"], metrics["config"]["test_seed"]}) == 3
    assert metrics["methods"]["latent_sigreg"]["per_fault"]
    assert load_file(out / "model.safetensors")
    assert (out / "replay.html").stat().st_size > 1000
    assert (out / "snapshot.png").read_bytes().startswith(b"\x89PNG")


def test_shift_baseline_uses_signed_motion_from_past_frames():
    pattern = torch.rand(32, 48, generator=torch.Generator().manual_seed(2))
    forward = torch.stack([pattern.roll(step, dims=-1) for step in range(5)])
    reversal = forward.clone()
    reversal[-1] = forward[-2].roll(-1, dims=-1)
    normal, fault = _shift_scores(torch.stack([forward, reversal])).tolist()
    assert normal[0] == 0
    assert fault[0] > 0.01
