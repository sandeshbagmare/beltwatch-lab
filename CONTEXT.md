# Project context

BeltWatch Lab tests whether a model trained on normal conveyor video can recognize unexpected changes. The footage is generated locally, the model runs on a CPU, and the replay compares it with simpler detectors.

The neural objective adapts LeWorldModel's next-embedding prediction and Gaussian regularization. The fixed-shift baseline performs best on the current simulator, and that result should stay visible in the documentation. This project has no real camera integration, personal footage, cloud inference, pretrained model downloads, or equipment controls.

Read the [project story](docs/ARTICLE.md) for the reasoning, the [method](docs/METHOD.md) for the implementation, and the [evaluation notes](docs/EVALUATION.md) before interpreting the numbers. The [data card](data/README.md) describes the generator. The original [design](docs/superpowers/specs/2026-11-20-beltwatch-design.md) and [interface plan](docs/superpowers/plans/2026-11-20-beltwatch.md) record the build, while [STATUS.md](STATUS.md) records verified work.

Version 2 adds a public GitHub Pages presentation and detector choices in the replay. It keeps the reference training recipe. The source that writes local reports also generates the replay published on the site. The browser plays saved inference scores from the Python run.
