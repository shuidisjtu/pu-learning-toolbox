## Purpose

<!-- Explain the problem, user impact, and scope of this PR. -->

## Main changes

<!-- Describe relevant API, algorithm, test, documentation, or benchmark changes. -->

## Validation

<!-- Record commands and results. Explain checks that could not be run or do not apply. -->

- [ ] `uv run pytest tests/ -m "not slow and not e2e"`
- [ ] `uv run python scripts/check_format.py` (ruff check + format --check)
- [ ] `uv run python scripts/check_comment_quality.py`
- [ ] `uv run python scripts/check_test_quality.py`
- [ ] `uv run python scripts/check_doc_links.py`
- [ ] `uv run python scripts/check_api_docs.py`
- [ ] `uv run python scripts/check_project_metadata.py`
- [ ] `uv run python scripts/check_math_rendering.py`
- [ ] `uv run python scripts/check_skill_sync.py`
- [ ] `uv run python scripts/check_baseline_configs.py`
- [ ] `uv run python scripts/check_survey_recipe_registry.py --require-present --require-locked`
- [ ] `uv run python scripts/generate_structure.py --check`
- [ ] `uv run python scripts/generate_layer_deps.py --check`
- [ ] `uv build`
- [ ] `git diff --check`

## Paper and experiment boundaries

- [ ] No paper algorithm or benchmark is involved, or the applicable items below are addressed
- [ ] Official / clean-room / paper-like status is stated accurately
- [ ] Configurations, seeds, data sources, and generation commands are traceable
- [ ] Test truth was not used to choose hyperparameters, class priors, or thresholds

### Departures from the paper or locked source, if applicable

<!-- Explain source behavior → project treatment → reason → validation and limits.
Document supported engineering choices here. Passing tests does not establish numerical
reproduction or validate data, recipes, or GPU paths that were not exercised. -->

### Decisions needed from collaborators (write "None" if resolved)

<!-- List unresolved methodology, protocol/budget, or evidence conflicts.
This section does not replace existing formal sign-off requirements. -->

## Compatibility and maintenance

- [ ] Public API documentation and registry metadata are consistent with the change
- [ ] Base imports work without optional dependencies
- [ ] README, architecture, project structure, and status records are updated as needed
- [ ] Public documentation and example comments use English; internal Chinese references are labeled
- [ ] Changed document links and heading fragments have been checked
- [ ] No credentials, local environment files, or unrelated generated artifacts are committed

## Remaining risks

<!-- State untested platforms, dependencies, data, GPU paths, and reproduction limits. -->
