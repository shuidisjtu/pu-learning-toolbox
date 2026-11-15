# Contributing

This guide covers code contributions, paper implementations, benchmarks, and project status.
External contributors can add algorithms and core features; the maintenance team leads core
architecture decisions.

## 1. Sources of authority

Use each source for its own subject:

| Subject | Source |
|---|---|
| Python, dependencies, extras, and tool configuration | [`pyproject.toml`](pyproject.toml) |
| Reproducible resolution for CI and local development | `uv.lock`; dependency declarations remain in `pyproject.toml` |
| Public signatures, parameters, returns, and errors | [API reference](docs/user/reference/api.md) |
| Module boundaries, dependency directions, and data flow | [Architecture](docs/dev/architecture.md) |
| File layout and workspace asset boundaries | [Project structure](docs/dev/project_structure.md) |
| Supported environments and CI responsibilities | [Compatibility](docs/dev/compatibility.md) |
| Algorithm interfaces, registration, and capability declarations | [New algorithm template](docs/dev/new_algorithm_template.md) |
| Source comments and their quality gates | [Comment governance](docs/dev/comment_governance.md) |

The five developer guides above, this guide, the user entry pages, quickstart, concepts, and
how-to guides and API reference are in English. Survey reproduction documents currently
remain in Chinese and will follow in the next batch. Internal plans and
historical decisions may remain in Chinese; they are optional background for ordinary contributions.
The English developer guides state the required contribution rules. Describe new architectural
tradeoffs in the PR for maintainers to resolve.

Maintainers record task and release status in
[`process_checklist.md`](docs/dev/process_checklist.md) and
[`release_process.md`](docs/dev/release_process.md) (Chinese). The
[ADR archive](docs/adr/README.md) (primarily Chinese) records decisions; the current
[language policy](docs/adr/0022-public-documentation-language.md) supersedes its historical
language rule. Method cards describe papers, formulas, implementation boundaries, and reproduction
specifications; a completed card alone does not establish implementation or numerical reproduction.

## 2. Development environment

CI covers Python 3.10, 3.11, and 3.12. Python 3.11 is recommended.

```bash
uv venv --python 3.11
uv sync --python 3.11 --extra dev --extra torch
```

Without uv, create and activate a virtual environment, then install the editable package:

```bash
python -m venv .venv
# POSIX shells:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,torch]"
```

Maintain installable dependencies only in `pyproject.toml`. PR CI uses the committed `uv.lock` for
deterministic resolution; nightly CI runs `uv sync --upgrade` in its temporary checkout to check
the newest resolvable versions within the declared ranges. `uv.lock` replaces the former
`requirements.txt`; do not reintroduce a parallel list. Add development tools to `dev` and runtime
dependencies to the appropriate extra.

## 3. Branches and commits

Develop each independent change on a feature or fix branch:

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/<short-name>
```

Use prefixes `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `refactor:`, or `chore:`. Keep each PR focused
on one concern. Exclude unrelated refactors, generated files, and local configuration. Never
commit private keys, tokens, `.env` files, or data access credentials.

For complex changes or changes spanning modules, push a feature branch and open a PR before
merging through the quality gates. Squash merging is recommended to give `main` one clear record
per task. The PR must explain the deliverable and validation without requiring a review of every
intermediate commit. If the client cannot create PRs, retain the pushed branch and provide a PR
creation link; do not push the change directly to `main`.

## 4. Public API rules

- Normalize and validate labels through shared public interfaces. Public PU labels use `{1, 0}`:
  positive and unlabeled, respectively.
- Classifiers implement `BasePUClassifier`; prior estimators implement `BasePriorEstimator`.
- Register new methods and declare assumption, scenario, backend, source status, implementation
  status, and capabilities as required by the [algorithm template](docs/dev/new_algorithm_template.md).
- Keep optional imports lazy so base imports work without extras. Missing dependencies must
  produce an actionable error when the feature is used.
- An `api_only` entry must not claim to be trainable. A clean-room implementation must not claim
  official numerical reproduction.
- Document and test public behavior. Breaking API changes require corresponding updates to
  architecture, examples, tests, and version notes.
- Follow the module boundaries and dependency directions in the English architecture guide.

## 5. Tests and quality gates

Run focused tests while developing. Add a regression test before fixing a bug and cover relevant
boundaries. Run the applicable checks before submitting; CI covers the complete platform matrix:

```bash
uv run pytest tests/ -m "not slow and not e2e"
uv run python scripts/check_format.py
uv run python scripts/check_test_quality.py
uv run python scripts/check_doc_links.py
uv run python scripts/check_api_docs.py
uv run python scripts/check_project_metadata.py
uv run python scripts/check_math_rendering.py
uv run python scripts/check_skill_sync.py
uv run python scripts/check_baseline_configs.py
uv run python scripts/check_survey_recipe_registry.py --require-present --require-locked
uv run python scripts/check_comment_quality.py
uv run python scripts/generate_structure.py --check
uv run python scripts/generate_layer_deps.py --check
uv build
git diff --check
```

| Marker | Purpose |
|---|---|
| `unit` | Local behavior, parameters, and boundaries |
| `math` | Hand-computable formula golden tests |
| `property` | Mathematical invariants |
| `contract` | Shared API contracts for native estimators |
| `integration` | Behavior across components |
| `paper` | Reproduction using paper data, source, or configurations |
| `slow` | Tests unsuitable for frequent local feedback |
| `e2e` | Complete workflows, run in the appropriate suite |
| `gpu` | Tests requiring suitable hardware |

Ordinary test files have at most 15 test methods and cover basic, parameter, edge, and deterministic
behavior. `scripts/check_test_quality.py` enforces these categories and documents explicit exemptions.
Its strict default is shared by local checks and CI. Slow, paper, E2E, and GPU validation needs
suitable data, time, or hardware; report what was actually run and any limits.

### 5.1 Shared helpers: reuse required

These helpers own concepts used across modules. Reuse them instead of introducing local copies.

| Helper | Location | Purpose |
|---|---|---|
| `canonical_hash` | `pu_toolbox/utils/serialization.py`; compatibility re-export in `benchmarks/_common.py` | Permissive canonical JSON hash with the default `allow_nan`, for report and manifest payloads |
| `strict_canonical_hash` | `pu_toolbox/utils/serialization.py` | Canonical JSON hash with `allow_nan=False`, rejecting non-finite values for artifact identities |
| `json_safe` | `pu_toolbox/utils/serialization.py` | Permissive report conversion: NaN/Inf to `None`, NumPy scalars to `item()`, and `Path` to `str` |
| `json_scalars` | `pu_toolbox/utils/serialization.py` | Strict manifest index conversion, rejecting non-scalars and non-finite floats before hashing |
| `array_hash` | `pu_toolbox/utils/serialization.py` | SHA-256 of dtype string, `json.dumps(shape)`, and contiguous array bytes; shape comes from the caller's array |
| `file_hash` | `pu_toolbox/utils/serialization.py` | SHA-256 of file bytes streamed in 1 MiB chunks, without a metadata prefix |
| `sigmoid_stable` | `pu_toolbox/utils/activations.py` | Numerically stable sigmoid |
| `rbf_weights` | `pu_toolbox/utils/basis.py` | Shared RBF kernel weights |
| `validate_true_binary_labels` | `pu_toolbox/core/validation.py` | Validation of the `y_true` value domain |
| `check_scalar_in_range` | `pu_toolbox/core/validation.py` | Scalar range validation; `inclusive=False` uses an open interval |
| `solve_prior_from_positive_fraction` / `stable_centroid_denominator` | `pu_toolbox/estimators/risk/_class_prior.py` | Prior derivation and stability checks for centroid term 1−2ph |
| `git_worktree_dirty` | `benchmarks/_common.py` | Dirty-worktree detection; `exclude` omits runner outputs |

Use `json_safe` for report output, where replacing NaN with `None` is intended. Use `json_scalars`
for index lists that enter artifact hashes, where invalid values must fail. Do not merge these
helpers. Likewise, keep `canonical_hash` for permissive report/manifest payloads and
`strict_canonical_hash` for identities such as `split_sha256`, `cache_key`, and `protocol_sha256`.
Their treatment of non-finite input is deliberately different.

Reuse `json_scalars` for array-to-JSON-scalar-list conversions that enter hashes. Two intentional
exceptions remain: `label_view_sha256` in `pu_toolbox/experiment/strategies.py` hashes a label view
already made integer with `.astype(int)`; report payloads in `pu_toolbox/diagnostics/uncertainty.py`
need permissive output semantics.

`_json_sha256` in `pu_toolbox/experiment/text.py` has a separate, intentional encoding:
`ensure_ascii=False`. Changing it changes non-ASCII corpus digests and cache filenames and is a
breaking change.

Keep `array_hash` and `file_hash` separate: the former describes an in-memory ndarray including
dtype and shape; the latter streams file bytes. Uploaded in-memory byte buffers belong to neither
interface. Existing public wrappers `survey_protocol.array_digest` and `split_archive.file_sha256`
delegate to the helpers; new consumers should use the helpers directly.

For a 0-d array, `array_hash` hashes shape `[]`, even though its internal contiguous copy has shape
`(1,)`. This was a deliberate compatibility correction to `survey_protocol.array_digest`; it did
not affect persisted artifacts. Coverage is in `tests/unit/utils/test_content_hashes.py`.

Other benchmark hashes encode previously recorded benchmark identities. Do not consolidate them
without freezing and validating their own compatibility boundary. The detailed inventory is
optional Chinese background in the [single-source map](docs/dev/single_source_map.md).

`json_safe` does not guarantee arbitrary input becomes serializable: it does not detect cycles
(which can raise `RecursionError`) and leaves unknown types such as `np.ndarray`, `set`, and `bytes`
unchanged.

Review incremental code for duplicated concepts. More than one single-source violation is a yellow
threshold: the PR must include consolidation. Three or more violations, or a third split of the
same concept, is a red threshold requiring a structural refactor assessment for that area. Public
interfaces need documentation and tests; refactors need covering tests and should address one issue
at a time. Record material design tradeoffs when making the decision. Historical audits are optional
Chinese background in [architecture principles](docs/dev/architecture_principles.md) and
[ADR-0001](docs/adr/0001-architecture-governance.md).

### 5.2 Comment rules

The English [comment governance guide](docs/dev/comment_governance.md) defines the full criteria,
gate behavior, and migration policy. In particular:

- Use contextual `TBD: ...` or `TBD(#123) ...` for unfinished work. Do not introduce `TODO`, `FIXME`,
  `XXX`, or `HACK` markers.
- Trailing comments are errors in migrated partitions under the default gate and discouraged
  advisories elsewhere. The gate script owns the partition list. `--strict-inline` is a manual
  migration switch. Tool directives such as `noqa`, `pragma:`, and `type:` are exempt.
- Repository paths referenced in comments, docstrings, and string literals must exist.
- Write comments and docstrings in English; legitimate localized runtime strings and test data
  can remain in their original language.

Run `uv run python scripts/check_comment_quality.py` (default scope: `pu_toolbox/`) and
`uv run python scripts/check_doc_links.py` (also checks source path references).

## 6. Paper methods and benchmarks

| Status | Meaning |
|---|---|
| Method card complete | Paper content and interface specifications are documented |
| Native/clean-room complete | An independent implementation runs and passes code tests |
| Paper-like complete | Experiments follow paper specifications but may use substitute data or a modern environment |
| Official reproduction complete | Official data, source/configuration, historical environment, and statistical protocol are locked and have actually been run |

Update benchmark runners, configurations, provenance locks, and results together. Submitted result
CSVs and manifests must be regenerable using repository configurations and commands. Document seed
counts, data sources, and environment limits in the relevant README.

Never use test-set truth to select class priors, thresholds, or hyperparameters. Synthetic `y_true`
and propensity are for evaluation or explicitly labeled oracle sensitivity only.

For departures from the paper or locked source, explain source behavior, project treatment, reason,
validation, and limits. Document supported engineering choices with the implementation. Passing
tests alone does not establish paper numerical reproduction. Request collaborator decisions for
unresolved methodological, protocol/budget, or evidence conflicts; do not sign off historical
reviews or bypass existing formal admission rules.

## 7. Documentation and project status

### Workspace assets and cleanup

Keep source separate from local experimental assets; the English
[project structure](docs/dev/project_structure.md) defines storage and script categories. Place new
uploads under the appropriate local `data/incoming/user-uploads/` category and deliverable tables
and notes under dated `data/outgoing/` directories. These local directories can be created as needed
and are not distributed by a clone. Do not stage large archives, private originals, or unverified
results with `git add -A`.

Before cleanup, inspect running jobs and references. Old protocols, comparison matrices, and
research evidence in source control are not caches. Preserve original hashes when archiving.
Provide a compatibility entry point or an explicit migration plan for changed paths. Delete only
verified reproducible artifacts with no unique evidence, list exact targets, and retain an operation
receipt. Never bulk-empty `data/`, `results/`, `tmp/`, or virtual environments. Use the experiment's
own reclamation mechanism for checkpoints.

### Status updates

Check user entry points and test counts in `README.md`, the `docs/README.md` index, API/module
boundaries in `docs/dev/architecture.md`, and the real layout in `docs/dev/project_structure.md`.
Check relevant task and release records in `docs/dev/process_checklist.md` (Chinese; maintainers
can update these from the evidence in your PR).

Mark a task complete only when its code, tests, documentation, and required experiments are complete.
Do not mark unrelated tasks done. Maintain public documentation in English at its existing paths
under the [language policy](docs/adr/0022-public-documentation-language.md); do not create parallel
language copies. During migration, state the completed scope accurately.

## 8. Pull requests

Push the feature branch, then open a PR:

```bash
git push -u origin feature/<short-name>
```

Describe the problem, implementation boundary, validation commands/results, experiment artifacts,
and unresolved risks. Follow the English [PR template](.github/pull_request_template.md). Merge only
after the Python CI matrix, static quality gates, and installed-wheel smoke tests pass.
