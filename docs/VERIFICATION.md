# Verification notes

Version 0.2.1 fixes input handling and the phone layout without changing the training recipe. A fresh full run reproduced every value in the reference metrics file.

## Checks in the repository

| Check | What it covers |
| --- | --- |
| 45 Python tests | Data isolation, paired fault labels, causal predictions, target gradients, regularization, metrics, thresholds, CLI output, report validation and published-page consistency. |
| 9 Chromium tests | All five detectors on every clip, selected frame decisions, keyboard controls, playback, narrow layouts, readable chart labels, high-density screens, downloads and offline playback. |
| Ruff and ty | Python formatting, lint rules and source type checks. |
| Source and wheel builds | Version metadata, included documentation and browser tests, and exclusion of cached bytecode. |

The browser suite serves the site locally under the same project path as GitHub Pages. It does not need the live website or any third-party page. It checks the saved score and threshold behind each displayed alert, rather than relying on screenshots alone.

CI runs the Python checks on Windows and Ubuntu. Ubuntu also installs Chromium and runs the browser suite. The source tests and browser tests can be run separately using the commands in the [README](../README.md#run-it-yourself).

## Bugs caught during this pass

The model API previously accepted NaN, infinity and unnormalized floating-point frames. It now rejects those values before fitting or scoring. The report writer also checks that frame buffers are grayscale uint8 arrays with the expected dimensions before exporting pixels.

The replay's original fixed-resolution canvas made axis labels shrink to about five CSS pixels when embedded on a narrow phone screen. The canvas now sizes its drawing buffer to the displayed chart and the screen's pixel density. Axis text is drawn at 12 CSS pixels; the browser test checks its rendered size after layout.

The review also corrected an outdated setup anchor, made the virtual-environment instructions explicit, refreshed the replay screenshot and excluded compiled Python files from source archives. Each behavioral fix has a regression check.

## Limits

These checks exercise generated footage and Chromium. They do not establish accuracy on real equipment, cover every browser, or prove that no defects remain. Benchmark results still come from one procedural generator and one reference training seed.
