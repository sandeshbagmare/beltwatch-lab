# BeltWatch Lab v2 implementation plan

**Goal:** Give visitors a working public demo and a readable account of the experiment.
**Spec:** [v2 design](../specs/2026-11-20-beltwatch-v2-design.md).

- [x] Report/CLI: test multi-method export and invalid inputs first, then add comparison scores and method selection; verify legacy report works. Regenerate replay from an unchanged reference benchmark and compare metrics to v1.
- [x] Writing: review public docs and add a candid project article, with accurate links, attribution, numbers and definitions.
- [x] Site: create static landing page with embedded replay, results, architecture explanation, paper references and reproduction instructions.
- [x] Verification: run Python checks/build, independent review, browser interactions and responsive/keyboard checks. Verify generated/static assets and links.
- [ ] Release: update version/status, commit/push intended files, enable Pages from docs, verify both CI jobs and the public page's actual content and replay.
