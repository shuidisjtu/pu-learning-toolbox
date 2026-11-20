# Using the command-line interface

The CLI retains independent tools. Training via `run` and `shift-run` is retired; see [migration](pipeline.md).

```bash
pu-toolbox make-demo-data --out-dir demo/ --n 200 --seed 42
pu-toolbox profile --data demo/X.csv --labels demo/y_pu.csv --out-dir results/
pu-toolbox list-methods
```

### Supporting commands

- `list-methods`: lists registered classifiers by name, family, prior requirement, input
  dimensionality, native architecture, main-input labels, and auto-instantiability.
  Capabilities come from registry metadata. Labels can be pu/pn/pnu; Auto-inst describes
  construction, not eligibility for PU workflows. `sensitivity` requires PU main-input semantics.
  Registered methods appear automatically.
- `list-priors`: lists available prior estimators. `km1` and `km2` map to
  `KernelMeanPriorEstimator(variant=...)`.
- `make-demo-data --out-dir demo/ [--n 200] [--c 0.5] [--n-features 5] [--separation 1.0] [--seed 42]`:
  uses `simulate_pu_dataset`. There are `n` samples per true class, 2n total. Training-readable
  outputs are `X.csv`, `y_pu.csv` and `sample_ids.csv`; `y_true.csv` is audit truth and
  `construction.json` records the source, random state and realized labeling budget.
  `--c` requests a positive-class labeling fraction under uniform fixed-count SCAR selection.
  The default separation 1.0 avoids the stronger prior-underestimation behavior of highly
  separated demo classes.
- `profile --data X.csv --labels y_pu.csv [--true-labels y_true.csv] [--out-dir .]`:
  profiles data and labeling-mechanism evidence, writing `profile.json`.
- `sensitivity --data X.csv --labels y_pu.csv [--classifier elkan_noto] [--class-priors 0.1,...,0.9] [--seed 42] [--out-dir .]`:
  scans class-prior/propensity assumptions and writes `sensitivity.json`.
- `audit-benchmark --result-dir results/ [--output audit.json]`: checks required artifacts,
  configuration hashes, trial/seed completeness, duplicate rows, finite metrics, and official-data
  PU split overlap and target-prior consistency. Failed checks exit with code 1;
  `paper_claim=false` and a dirty worktree remain warnings.
- `shift-audit --source-data source.csv --source-labels source_labels.csv --target-data target.csv [--target-labels target_labels.csv] --out-dir shift_results/`:
  uses an OOF domain classifier to audit source/target differences. Writes `shift_report.json`,
  `shift_report.md`, and `source_importance_weights.csv`. Detection and adaptation are separate;
  marginal weights have covariate-shift guarantees. See [distribution shift](distribution_shift.md).
- `shift-monitor --reference-data source.csv --reference-labels source_labels.csv --window-data current.csv [--window-labels current_labels.csv] --window-id 2026-08 [--history previous_history.json] --out-dir monitor/`:
  appends a deployment window and writes alerts, a window shift report, source weights,
  and resumable `shift_history.json`.
- `review --model trusted_model.pkl --data current.csv [--labels current_labels.csv] --query-budget 20 --query-strategy uncertainty --out-dir review/`:
  produces selective predictions and an active human-review queue. Load only models you created
  or trust; do not load unknown pickle files.
- `skill install [--force] [--dest agent-home/]`: installs the pu-workflow entry and six
  reference files into `~/.claude/skills/` and `~/.agents/skills/`. `--dest` changes their
  shared root. Existing installations are retained unless `--force` is used.
  The skill chooses CLI or Python APIs by task; see [the skill guide](using_skill.md).


See command `--help` for current options. CSV inputs require headers; PU label 0 means unlabeled, not known negative.
