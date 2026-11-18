# Enabling and using the pu-workflow skill

> Prerequisites: Python 3.10+, `pu-toolbox`, and a client supporting
> [Agent Skills](https://agentskills.io). Prefer `uv` in a project environment.
> Install the torch extra for image and encoder workflows.

The skill chooses toolbox CLI commands or Python APIs from a natural-language task. It covers
ordinary PU analysis, data construction, prior resolution, tuning/comparison, four-role studies,
PA/OA selection, image representations, shift, monitoring, human review, and benchmark audits.
The entry loads six reference files by task and interprets actual saved artifacts.

The bundled skill metadata declares version 1.11.0. Its instructions require `pu-toolbox>=1.11.0`
and select routes using the installed command help, Python signatures, and capability declarations.
Current-source research capabilities should be verified in the installed build before use.

## 1. Install and enable

For an installed package, run in your own environment:

```bash
uv pip install "pu-toolbox>=1.11.0"
pu-toolbox skill install
```

In a source checkout, use the current source:

```bash
uv sync --python 3.11 --extra dev --extra torch
uv run pu-toolbox skill install --dest work/agent-home
```

The installer copies the entry and reference files to two targets:

- `~/.claude/skills/pu-workflow/`: Claude Code / Cursor.
- `~/.agents/skills/pu-workflow/`: Codex / Gemini CLI / Windsurf.

`--dest` sets the common root for those two relative directories. Existing installations are
retained by default. To update an installed copy, use `pu-toolbox skill install --force`.
Clients can also discover the repository's skill resources directly.

The main instructions and technical references are English. The skill responds in the user's
language, including Chinese; its localized interpretation resource remains available.
Documentation translation alone does not require reinstalling or changing the skill version.

## 2. Describe the task

Supply paths, goals, prior information, and execution budget. For example:

> Analyze `X.csv` and `y_pu.csv`. They come from one population pool, with known population
> prior 0.4. Use CPU, 3-fold CV, seed 42, and save reports under `work/analysis`.

> I have fully labeled data. Split by original source, construct fixed-count SCAR and
> feature-dependent SAR PU observations, compare the settings, and save source indices
> and sampling records.

> Run PA and OA on four-role data, select checkpoints and thresholds independently,
> and then evaluate test.

These task descriptions can also be given in Chinese or another supported conversational language.

| Task | Route and artifacts |
|---|---|
| Ordinary OS tabular analysis | CLI/Pipeline: profiles, explicit classifiers, priors, and training reports |
| Independent TS and explicit prior targets | Python `input_sampling="ts"`: target conversions and per-fold consumption |
| OS/TS and SCAR/SAR construction | Preprocessing APIs: PU labels, sampling policies, provenance, and prior metadata |
| Tuning or multi-method comparison | `PUTuner` / `PUModelComparator` or CLI configuration: candidates and selections |
| Four-role studies and PA/OA | `ExperimentRunner`: separate selections, checkpoints, thresholds, and test results |
| Images, encoders, and frozen features | Capability-compatible models and representation paths: model/representation identities |
| Shift, monitoring, review, and audits | Corresponding CLI commands: structured diagnostics and execution artifacts |

The skill reuses existing user choices and authorization. Diagnostic findings inform method
choice and reporting. The client asks concrete questions when missing research design, provenance,
or extra budget affects the work. For a complete ordinary analysis, it can choose an appropriate
prior estimator when required and record its provenance.

## 3. Sampling, priors, and images

The current `profile` and `run` CLI commands use OS input semantics.
Independent TS uses Python APIs with an explicit sampling design; see the
[pipeline guide](pipeline.md) and [sampling tutorial](pu_sampling.md).

Pipeline's call-level `class_prior` and report `prior.value` target the population. Reports
retain the estimator's original target, population conversion, and actual training consumption
per fold. Select the classifier explicitly and use the Pipeline to resolve the prior chain. See
[prior semantics](../concepts/class_prior_semantics.md).

For raw images, use finite NCHW arrays and an explicit compatible CNN classifier, for example:

```bash
pu-toolbox run --data X.npy --labels y_pu.csv --classifier nnpu --architecture cnn --class-prior 0.4 --device cpu --max-epochs 2 --evaluation-cv 2 --out-dir work/images
```

For custom encoder and frozen-feature lifecycles, see
[representation concepts](../concepts/representations.md) and the
[encoder tutorial](custom_encoders.md). For four-role data and selection records, see
[custom experiments](custom_experiments.md).

## 4. Installed references

Each installation target contains the same seven resources:

| Resource | Content |
|---|---|
| `SKILL.md` | Task recognition, capability discovery, routing, and completion conditions |
| `references/analysis.md` | Ordinary analysis, tuning/comparison, and sensitivity |
| `references/sampling-and-priors.md` | Sampling, labeling, prior targets, and consumption chain |
| `references/research-experiments.md` | Four roles, PA/OA, checkpoints, and custom strategies |
| `references/representations.md` | CNNs, encoder injection, and frozen features |
| `references/extensions.md` | Shift, monitoring, review, and audits |
| `references/interpret.zh-CN.md` | Chinese field interpretation and conclusion writing |

References resolve relative to the skill directory. Installed users can work in their own data
directory using the included minimal examples. Source users can also run
[examples 14–18](../../../examples/minimal/).

## 5. Results and troubleshooting

The client reads actual JSON artifacts and reports data settings, methods, prior provenance and
consumption, available metrics, and next actions. Pipeline schema 1.1 includes
`prior.training_consumption` and valid-fold counts. Research manifests retain separate PA/OA
selections and final test results. Interpretations should state evidence-supported conclusions
and their applicable conditions.

| Situation | Action |
|---|---|
| The client does not activate the skill | Check installation paths, client discovery settings, and task wording; invoke `$pu-workflow` explicitly |
| An installation already exists | Retain it or deliberately update with `--force` |
| Commands or Python symbols differ from the route | Check installed version, command help, and signatures; install a build with the required capabilities |
| Input errors | Correct the reported format, dimensionality, labels, or parameters, then rerun |
| Diagnostic warnings or unavailable metrics | Read the specific fields/actions, continue the authorized route, and report needed data actions |

Maintainers run `uv run python scripts/check_skill_sync.py` to check complete resource trees,
internal links, and portable frontmatter. Installation tests, wheel contents, and execution
outside the checkout validate distribution separately.
