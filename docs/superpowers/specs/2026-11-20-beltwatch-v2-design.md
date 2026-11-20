# BeltWatch Lab v2

The first release runs locally and publishes a downloadable replay. This release should let a visitor understand and use the experiment directly from GitHub Pages. The user also asked for a full writing review. Decisions here are the implementer's choices under the user's instruction to proceed without questions.

## Deliverables

1. A static public page served from the main branch's docs folder. Include a working embedded replay, a clear explanation of the experiment, the real results including the winning simple baseline, source papers, and reproduction instructions. Keep the page usable on phones and with a keyboard; load all assets locally.
2. Add a detector selector to the replay. All five methods should have their own aligned trace, threshold, metrics and explanation. A run without optional comparison scores should still produce the original single-method report.
3. Review every public Markdown article and guide for plain, natural writing. Preserve metrics, attribution and limitations. Keep internal plans as a historical record while adding this release's plan and evidence.
4. Preserve training and data generation. A regenerated reference run must match the existing metrics. Do not tune against the final holdout.
5. Package as version 0.2.0. Verify local tests, lint/types, build, replay interaction, desktop/mobile page behavior, links, and public deployment. Push the complete reviewed result and verify both CI and the live GitHub Pages site.

## Report contract

Extend write_report with keyword-only optional comparison_scores: Mapping[str, Sequence[NDArray[np.float64]]] | None. Each mapping key is a known metric method, and every sequence is aligned to examples. All supplied arrays have length T - CONTEXT, finite nonnegative entries. The original examples' arrays remain the primary latent_sigreg score. Preserve JSON clips[*].scores for compatibility and add method_scores to each clip. UI dropdown shows methods for which all clips have data.

## Publication

Use the already authorized public repository sandeshbagmare/beltwatch-lab and main branch. Inspect existing Pages settings before creating or updating them. Configure branch Pages at /docs after sources are committed/pushed. Never publish local runs, virtual environments or credential helpers.
