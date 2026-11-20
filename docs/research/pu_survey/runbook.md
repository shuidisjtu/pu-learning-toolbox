# Running Experiments and Generating Reports

This guide covers the repository's Survey example scripts. To call `ExperimentRunner`
directly, prepare four data roles, or extend generators, trainers, and selection strategies,
see [custom experiments](../../user/howto/custom_experiments.md).

Run commands from the repository root with Python 3.11:

```powershell
uv sync --python 3.11 --extra dev --extra torch --extra report
```

New runs use `survey-v3.2` (revision `2026-11-08-clean-support-coverage`) and
`survey-recipes-v1.3`. Summarize historical v3 artifacts read-only under their original
records; do not reinterpret them as new runs.

## Prepare four data roles

`ExperimentRunner` accepts caller-prepared train, pu_val, clean_val, and test partitions.
The toolbox also supplies a preparation script:

```powershell
uv run python scripts/prepare_survey_splits.py --datasets spambase --seeds 0,1,2,3,4
```

By default it reads `data/raw` and writes `data/splits/<dataset>/split_<seed>/`.
Each split contains `train.npz`, `pu_val.npz`, `clean_val.npz`, `test.npz`, and
`split_manifest.json`. NPZ files contain true labels and disjoint global indices;
the generator constructs PU labels during execution. Read the prior from the split's
`class_prior.population`, without deriving it from test truth.

PULNS, LaGAM, and full-meta Self-PU use independent clean support. For the five original
tabular/image datasets, rebuild it through the reviewed raw reader. For IMDB, 20News,
and the `adni` substitute-data slot, carve directly from a frozen split:

```powershell
# Original five datasets: reviewed raw reader.
uv run python scripts/prepare_clean_support.py --dataset spambase --seed 0 --support-size 100 --group-policy not_applicable --raw-dir data/raw --out-dir data/clean-support/spambase/split_0 --write

# Text: SBERT features from a frozen split.
uv run python scripts/prepare_clean_support.py --dataset imdb --split-dir data/splits/imdb/split_0 --seed 0 --support-size 500 --group-policy not_applicable --out-dir data/clean-support/imdb/split_0 --write

# ADNI slot: frozen raw images; explicitly record the subject-isolation gap.
uv run python scripts/prepare_clean_support.py --dataset adni --split-dir data/splits/adni/split_0 --seed 0 --support-size 500 --group-policy unverifiable --owner-approval approved_by_protocol_owner_20261108 --out-dir data/clean-support/adni/split_0 --write
```

Support is disjoint from the remaining four roles. The configured support size is 100 for
Spambase and 500 for other datasets. `--support-seed` must equal `--seed`; omission uses
`--seed`, and an explicit mismatch is rejected. The output directory must not exist;
omit `--write` to preview. For the ADNI slot, `unverifiable` records that subject isolation
cannot be checked; it does not establish isolation. The approval token above records a
specific protocol-owner decision, not general authorization for another study. See the
[data-source and interpretation limits](data/README.md).

For `self_pu_full_meta`, omit `--os-or-ts` or set `--os-or-ts os`. The single-unit
and batch paths reject `ts` for this versioned profile. Report its OS clean-support
results separately from the TS-compatible `self_pu` ablation; see
[protocol §2.3](pu_survey_protocol.md#23-pu-data-generation).

## Run the supplied Survey example

Validate the current recipes:

```powershell
uv run python scripts/check_survey_recipe_registry.py --require-present --require-locked
```

Run one unit using a prepared split:

```powershell
uv run python scripts/run_survey_experiment.py data/splits/spambase/split_0 --protocol survey-v3.2 --recipe-registry pu_toolbox/experiment/survey_recipe_registry_v1_3.json --dataset spambase --method upu --training-path native_2d --labeling-mechanism scar --seeds 0 --c 0.1 --class-prior 0.3940447729 --device cpu --out-dir results/example
```

The prior above illustrates the current Spambase setting; use your split's own
`class_prior.population`. Supply it explicitly to this prior-requiring single-unit method;
the batch entry reads recorded split priors. For uPU, PA uses this population prior in both
views. The Survey default (`ts`-compatible) also fits with it; explicit `--os-or-ts os` fits
with the native-U prior derived from the training split's realized labeled fraction.
See `research_settings.prior` in the manifest for both values and sources.
For a PN oracle, replace `--method upu` with
`--oracle` and remove `--class-prior`; selection is OA only. Support methods require the
corresponding `--support-dir`. The audited product keeps clean support as a fifth role,
with original sample IDs disjoint from `train`, `pu_val`, `clean_val`, and `test`. The
unit command verifies the product and its match to the actual split before training or
evaluation. Each seed must use its own split; do not pass `split_0` as input for multiple seeds.

Use the batch entry for deterministic grid execution and resumption:

```powershell
uv run python scripts/run_survey_pilot.py --protocol survey-v3.2 --recipe-registry pu_toolbox/experiment/survey_recipe_registry_v1_3.json --splits data/splits --results results/example-grid --datasets spambase --methods upu --device cpu --dry-run
```

Review the plan before removing `--dry-run`. Narrow the scope with `--datasets`,
`--methods`, and `--training-paths`; `--plan-json` saves the dry-run plan. Repeating the same
request skips runs whose manifests satisfy the completion and identity checks.
`--keep-going` records a failure and continues. Methods requiring support need
`--support-root`. The batch driver checks all five role files and their sample IDs
before planning or resuming runs.

Batch runs store shared code/dependency evidence under `<results>/batch_evidence`, referenced
by each manifest; single-unit execution records those identities directly. Retain shared
evidence and selected weights when moving results. Reading results does not require the
original machine's absolute paths to exist.

For four-role data construction, parameter changes, and strategy extension through Python,
see [custom experiments](../../user/howto/custom_experiments.md). Exact signatures are in the
[API reference](../../user/reference/api/tools.md#four-role-execution-and-strategy-injection).

## Summarize and export results

New manifests record sampling, labeling policy, risk pool, training/selection priors,
validation supervision, and representation in `research_settings`. `summary.json` retains
the full settings per seed; `summary.md/csv`, workbooks, and diagnostic cards show summaries.
Historical results are identified as records read under their original manifest settings
and retain their original protocol identities. Field contracts are in the
[API reference](../../user/reference/api/tools.md#result-reading-statistics-and-export-results--reporting).

Summarize manifest files or directories directly, writing to a separate output directory:

```powershell
uv run python scripts/summarize_survey_results.py --manifests results/example-grid --out-dir results/report
uv run python scripts/export_survey_results.py --summary results/report/summary.json --out-dir results/report
```

Alternatively, run both stages with one command:

```powershell
uv run python scripts/export_survey_results.py --manifests results/example-grid --out-dir results/report
```

Multiple input paths are accepted; repeated paths are read once. Different files claiming
the same result identity are rejected to prevent silent replacement. Expected seeds default
to 0–4; use `--expected-seeds 0` for single-seed verification, retaining the single observation.
Mark pending and excluded paths with `--pending <path...>` and `--exclude <path...>`;
they remain in sources/coverage without contributing to means. When only a summary is
available, its aggregated sources cannot be separated again; check its original review status.

Summarization writes `summary.json/md/csv`, `coverage.csv`, and `refusals.json`. JSON is
authoritative for complete details, costs, sources, selections, and reproduction identities.
Export creates a workbook with four result-sheet groups, diagnostic cards, and conclusions.
Missing seeds remain visible in coverage rather than being filled to five. Undefined additional
metrics retain available counts and reasons; PA/OA rows do not double-count shared fit costs.

Optional `--literature facts.json` accepts `{"facts": [...]}`; each fact needs `fact_id`,
`method`, `statement`, `signals`, and `source`. Optional `--native-views views.json` accepts
a method-to-os/ts/both mapping such as `{"nnpu": "ts"}` to interpret native assumptions,
without changing the actual run_view. Missing metadata does not justify inferring literature
agreement or native-view groups.

Reading and export use only information actually recorded in the inputs. They do not retrain,
backfill original manifests, or automatically approve pending results.
