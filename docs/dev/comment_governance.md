# Source Comment Guidelines

> This document defines what belongs in source comments and how the gates enforce the rules.
> Optional background in Chinese: [ADR-0019](../adr/0019-comment-quality-gate.md) explains the
> tradeoffs; [architecture principles](architecture_principles.md) discuss broader maintenance
> practices and signs of architectural decay.

## 1. Evaluation criteria

A comment's value depends on whether it conveys information that **the code cannot express**.
Prefer descriptive names, decomposition, constants, and simple control flow. Avoid comments that
repeat what the code already says.

Keep three kinds of comments:

| Kind | Purpose |
|---|---|
| Intent | Why this code exists and what goal it serves |
| Overview | The overall structure of a long function or module |
| Information the code cannot express | Constraints, tradeoffs, sources, risks, numerical stability, and invariants |

Put other information in its authoritative location:

| Information | Authority |
|---|---|
| Module layers, dependency directions, data flow | [Architecture](architecture.md) |
| File and directory responsibilities | [Project structure](project_structure.md) |
| Public API signatures and semantics | [API reference](../user/reference/api.md) |
| Paper formulas, sources, reproduction status | [Method cards](../research/method_cards/) (Chinese research background) |
| Design tradeoffs and alternatives | [ADRs](../adr/README.md) (Chinese historical background) |

## 2. Mechanical gates

| Gate | Checks | Default severity |
|---|---|---|
| `scripts/check_comment_quality.py` | Legacy markers, bare TBD, trailing comments | Error / error / advisory, with partition-specific trailing-comment errors |
| Rule 1 of `scripts/check_doc_links.py` | Existence of paths enclosed in backticks | Error |

The comment gate also reports `scan-error`: a Python file that cannot be scanned because of
syntax, tokenization, or non-UTF-8 encoding errors produces an error finding. Scanning zero files
fails with exit code 1. A nonexistent path or empty scan set cannot count as a successful check.

### 2.1 Legacy markers

Use `TBD` for unfinished work instead of `TODO`, `FIXME`, `XXX`, or `HACK`.
Legacy-marker matching uses word boundaries and is case-insensitive.

### 2.2 Bare TBD

A `TBD` marker must be followed by `:` or `(`, allowing whitespace in between:
`TBD: ...`, `TBD : ...`, or `TBD(#123) ...`.

This syntax is only a proxy for traceable context. Reviewers must still assess whether the
context is sufficient. `TBD` matching is case-sensitive; lowercase `tbd` is ordinary prose
and does not claim to be a marker.

### 2.3 Trailing comments

Trailing comments are discouraged. Tooling directives are exempt: `noqa`, `pragma:`,
`type:`, `fmt:`, `ruff:`, and `isort:`. Other trailing comments produce advisories
unless the applicable policy makes them errors. During a directory migration,
`--strict-inline` promotes them to blocking findings.

The default strictness is defined **by partition** in `scripts/check_comment_quality.py`,
the single source of truth for this policy. CI uses these partitions automatically;
`--strict-inline` remains a manual migration switch. Trailing comments in migrated partitions
are errors; those in pending partitions remain advisory. A passing run reports the pending
partitions, so this document does not pin their changing counts.

A coverage assertion rejects a new tracked top-level package entry that belongs to no partition.
Each new entry must therefore be classified explicitly.

**Known boundaries:**

- Prose following a tooling-directive prefix, such as `# type: business rationale`, cannot
  reliably be distinguished from a real directive by this mechanism.
- The coverage set comes from the Git index. A new top-level directory is outside the assertion
  until it is staged with `git add`.
- Passing explicit paths disables partition strictness for that run. Those paths receive
  advisories unless `--strict-inline` is supplied, which makes the entire scan strict.

### 2.4 Path existence

A path enclosed in backticks must exist if it starts with an allowlisted root
(`pu_toolbox`, `tests`, `scripts`, `examples`, `docs`, or `external`) and ends in
`.py` or `.md`. Other roots, including `benchmarks` and `.github`, are outside this check.

Source files are scanned as complete text. Backtick paths in ordinary string literals are
therefore checked too, alongside comments and docstrings; this is a known boundary of the rule.

Resolve a finding by correcting the path, or remove backticks when the text is not asserting
the existence of a repository file.

Two deliberate exclusions apply:

- Shorthand tokens containing `{`, `}`, `*`, or `?` do not claim to identify one file.
- `tests/` is excluded from the scanned corpus because invalid paths there are negative
  fixtures for the gate.

## 3. Comments deliberately retained

The following comments deliberately remain inline. They carry information that names or types
cannot express, and placing a short annotation beside its line makes it easier to compare.
Do not delete them as redundant during cleanup.

| Location | Retained information | Why code alone is insufficient |
|---|---|---|
| `pu_toolbox/core/config.py` | Intervals for both clipping epsilons | Endpoint inclusion differs intentionally: `(0, 1)` and `(0, 1]` |
| `pu_toolbox/core/config.py` | Warning semantics of `MAX_PU_RATIO` | Severity: `validation.py` uses `warnings.warn` rather than blocking |
| `pu_toolbox/core/config.py` | Raising semantics of `MIN_POSITIVE_SAMPLES` | Severity: `validation.py` raises immediately |
| `pu_toolbox/core/tags.py` | Probability definitions of `SCAR` and `SAR` | The formulas define these values |
| `pu_toolbox/core/tags.py` | Criteria and numerical boundary for all four `TrainingCost` levels | Closed-form/convex, bounded iteration, fixed-epoch nonconvex, and unclassified neutral scoring; the boundary is approximately 1000 epochs |
| `pu_toolbox/core/tags.py` | Meanings of `API_ONLY` and `NATIVE` | The former means no training logic; the latter identifies a clean-room implementation |
| `pu_toolbox/losses/llsvm.py` | Thirteen shape annotations | Plain `np.ndarray` signatures carry no shape information; the project does not use shape type annotations |

Section 3 retains 25 lines (**machine-checked anchor**:
`tests/unit/scripts/test_check_comment_quality.py` compares the current trailing-comment counts
with these values): `pu_toolbox/core/config.py` 4,
`pu_toolbox/core/tags.py` 8, `pu_toolbox/losses/llsvm.py` 13.

A separate repository-wide style decision is needed for the following:

- The nine `# ── X ──` section headings in `pu_toolbox/core/tags.py` repeat the class names
  immediately below them (four match exactly; the others differ only in case or spacing).
  The criteria in section 1 would remove them, but this style is widespread across modules.
  Changing one file alone would make the repository inconsistent, so this is outside a
  comment-cleanup batch.

## 4. Migration strategy

1. First make the gate capable of failing, then clean existing comments.
2. Migrate trailing comments one module at a time; use `--strict-inline` to check the directory
   being migrated.
3. Classify a comment before deleting it. Remove information already expressed by code;
   move other information into an intent comment before the block.
4. Comment edits must not change behavior, random sequences, serialized fields, or public APIs.
5. A false positive requires a minimal reproduction test. Fix the gate rather than adding a
   permanent exemption.
6. Keep `core/` and `losses/` pending for now. Whole-directory strictness would reject the
   25 deliberately retained comments in section 3. Choosing a retention-exception table,
   rewriting them as preceding comments, or retaining advisory severity is a separate decision.

## 5. Language

The current policy is [ADR-0022](../adr/0022-public-documentation-language.md).

- Use English for source comments and docstrings.
- Public user documentation, required external-contributor documentation, and public PU Survey
  reproduction documentation use English, maintained at their existing paths.
- Internal plans, research notes, and historical ADRs may remain in Chinese; localized
  interpretation resources retain their intended language.
- Preserve original identifiers for formulas, protocols, classes, fields, and document paths.
