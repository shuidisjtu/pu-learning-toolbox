# ADR 0023: Explicit method selection

- Status: Accepted
- Date: 2026-11-18

## Context

Metadata-based rankings and an implicit pipeline default obscure the caller's method choice
and can be mistaken for empirical model selection. Explicit candidate comparisons retain
validation evidence and expose assumptions and supervision.

## Decision

Retire advisor rankings, `recommend_methods` and `recommend_from_profile` on every public
path, the CLI `recommend` command, and both implicit and explicit `classifier="auto"`.
Keep `PUPipeline` with an explicitly supplied classifier, registry discovery, profiling,
prior estimation, diagnostics, and the experiment layer. Keep `PUTuner` and
`PUModelComparator` for explicitly supplied candidates. Do not add a compatibility layer.

## Alternatives and consequences

Keeping a deprecated automatic path or replacing it with another ranking would preserve
the ambiguity; neither is adopted. Callers must select methods using registered capabilities,
assumptions, prior and supervision requirements, and permitted validation. Existing clients
will require migration. Historical records remain historical; this decision supersedes any
earlier description of advisor ranking or automatic classifier selection as the current
architecture. The core advisor package and its public exports have been removed.
