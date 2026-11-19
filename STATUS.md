# Current status

| Owner | Workstream | State |
| --- | --- | --- |
| Integrator | Scope, docs, packaging, evaluation, CLI, release | Published |
| Simulator | Synthetic conveyor and split integrity | Verified |
| Model | CPU latent predictor and regularizer | Verified |
| Demo | Static replay and snapshot rendering | Verified |
| QA and security | Independent verification | Reviewed |

## Evidence log

- Integrator: Reviewed the GitHub portfolio and scoped a new, standalone repo. Selected LeWorldModel's predictive objective and SIGReg for an action-free synthetic conveyor experiment.
- Simulator: Added NumPy-generated conveyor clips with isolated splits and matched jam, reverse and debris test variants; documented generation and limitations in data/README.md. RED: tests/test_data.py could not import the missing simulator module. GREEN: 3 simulator tests passed in the Python 3.12 project environment and Ruff lint/format passed. A four-seed sweep verified all 1,536 fault pairs begin changing on the labeled frame with no changes outside the belt. Full-suite collection reached a local Windows Application Control block on the uv Python _multiprocessing DLL when importing PyTorch; routed to the integrator.
- Model: Implemented the per-frame CNN and four-frame causal latent forecast, a differentiable projected Epps-Pulley regularizer, and a seed-matched zero-weight ablation. RED: model tests stopped on the missing beltwatch.model module under Python 3.14. GREEN: all 5 model tests passed, the integrated suite passed all 17 tests, and Ruff lint/format plus ty passed on the model and its tests.
- Demo: Added a single-file replay and a Pillow snapshot with aligned clip pixels, latent scores, event labels, a shared scale and the normal calibration threshold. RED: report tests stopped on the missing beltwatch.report module. GREEN: 2 report tests passed; all 17 suite tests passed on the Python 3.14 environment, and targeted Ruff lint/format and ty passed. Edge/Playwright verified keyboard seeking, the first score at frame 4, fault and alert display, play/pause, switching clips, mobile width, and only the loopback request for the page.
- Independent benchmark QA: Validated split isolation and first changed frame across 144 paired faults, compared tied-score AP/AUROC with direct reference calculations for 200 randomized cases, verified the 7/73 leave-one-out exceedance rate for nominal calibration, and confirmed overwrite protection preserves existing files. Seed 2026 was a development pilot; a persistence residual plus training-only within-clip centering of SIGReg restored latent spread (1.10 versus 0.0018 with the otherwise matched uncentered pilot), with 80.7% event recall and 8.0% false alarms on the pilot. Do not use pilot numbers as final results.
- Defensive review: No blocking code or workflow findings. Verified inert HTML data and no extra browser requests, caller-selected new output directory, read-only workflow permissions, and pinned actions. A runtime dependency advisory audit was clear after upgrading setuptools; the CPU wheel uses the same base version separately checked against PyPI advisories.
- Integrator: Switched to the installed Python 3.14 after isolating the portable 3.12 loader issue. The reference architecture uses per-frame group normalization and a persistence residual; a regression catches the near-zero-spread pilot. Added a warm-up-fitted pixel-shift baseline and renewed calibration/test draws before the reference run. Full local run finished in ~37 seconds; 19 tests (95% line coverage), Ruff, ty, package build and browser replay checks passed. See docs/assets/metrics.json.
- Release: Verified the public repository's default branch is `main`. A second full run reproduced the reference metrics, replay HTML and figure byte for byte. [GitHub Actions](https://github.com/sandeshbagmare/beltwatch-lab/actions) tracks Ubuntu and Windows checks.
