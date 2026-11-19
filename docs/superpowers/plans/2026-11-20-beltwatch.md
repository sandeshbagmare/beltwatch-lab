# BeltWatch Lab Implementation Plan

> **For agentic workers:** Use subagent-driven-development or executing-plans. Report targeted red/green checks in STATUS.md.

**Goal:** Train, calibrate, compare and replay a CPU-only synthetic conveyor anomaly detector.
**Architecture:** Independent simulator, predictive model, metric/orchestration and offline demo modules.
**Tech stack:** Python 3.12+, PyTorch CPU, NumPy, Pillow, safetensors, pytest.
**Spec:** [BeltWatch Lab design](../specs/2026-11-20-beltwatch-design.md).

## Contract

All code is under `src/beltwatch`; tests are under `tests`. The integrator sets up `.venv`; for focused tests during setup, use `$env:PYTHONPATH='src'` and the available interpreter. CPU path only; if torch is missing wait for the shared environment rather than installing globally.

- [x] Simulator: `data.py` provides `CONTEXT = 4` (the number of causal context frames), `FAULTS: tuple[str, ...]` (at least three types; jam, reverse, and one appearance fault), `@dataclass(frozen=True) class Clip` with fields `clip_id: str, scene_id: int, fault: str, frames: NDArray[np.uint8], labels: NDArray[np.bool_]` (frames [T,32,48], T default about 32; labels [T]; fault is "normal" or a member of FAULTS). Function `make_split(split: Literal["train", "calibration", "test"], n_scenes: int, *, seed: int = 2026) -> list[Clip]`. Train/calibration are normal only; test has one normal and one fault of each kind per scenario, all sharing identical pre-event frames per pair. Encode split in scene/seed identity. Fault labels correspond to first changed dynamics/frame (no reward/action metadata). Cover determinism, split disjointness, first changed index, and test pairing. Also own data/README.md.
- [x] Model: `model.py` exposes `BeltModel()` (fixed architecture), `frame_scores(model: BeltModel, clips: Tensor) -> Tensor` input CPU float [B,T,32,48], normalized [0,1], output [B,T-CONTEXT] score aligned to targets t>=4. Expose `fit_model(clips: Tensor, *, epochs: int, seed: int, sigreg_weight: float = 1.0) -> tuple[BeltModel, list[dict[str, float]]]`; log mean prediction, SIGReg and total loss per epoch as keys `prediction`, `sigreg`, `total`. Both branches receive gradients. Implement projected Epps–Pulley quadrature and no-SIGReg ablation via zero weight. Expose `latent_spread(model: BeltModel, clips: Tensor) -> float` (mean across per-channel standard deviations as collapse diagnostic). Test causal alignment, deterministic CPU training, projected penalty, and target gradient. Import CONTEXT from data.
- [x] Demo: `report.py` exposes `write_report(directory: Path, examples: Sequence[tuple[Clip, NDArray[np.float64]]], metrics: Mapping[str, Any]) -> None`. Input example scores length T-CONTEXT; metrics schema `{"methods": {name: {"clip_ap": float, "clip_auroc": float, "frame_ap": float, "recall": float, "false_alarm_rate": float, "threshold": float, ...}}, "config": {...}}`, primary method key `latent_sigreg`. Emit self-contained `replay.html` with built-in clips, keyboard-usable controls and score timeline, and `snapshot.png` using Pillow (a diagram/figure rendered from actual held-out data). No CDN or inline executable interpolation from JSON; test report generation and context offset. Demo only sees generated clips.
- [x] Integrator: `metrics.py` handles exact tie-aware AP/AUROC, clip calibration on normal peaks and comparative metrics; `cli.py` builds split, trains full and ablation, computes baselines, aggregates, serializes JSON and safetensors, runs demo. Make `beltwatch run --out <new-dir>` and `--quick` (smaller smoke run), validate input and avoid overwriting existing directories. Package and document a CPU install; add Windows/Linux CI. Own integration and metrics tests.
- [x] Independent QA and defensive code review: run real CLI and tests, check isolation, causality/report alignment, artifact handling, offline behavior, and dependency/workflow hygiene. No exploit development.
- [ ] Publish verified sources and selected reproducible artifacts to the new repo's main branch and verify remote state.

## Review focus

1. Adjacent frames or paired test variants cannot leak across splits.
2. The forecast for time t cannot read pixels at t or later; score t may compare against the observed t.
3. Calibration cannot see test faults or scores, and ties use a consistent decision rule.
4. Output to an existing directory must fail without mutating its contents.
5. Report labels and plots must align with target frame indices; score traces begin after context.
