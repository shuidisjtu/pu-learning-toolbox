# ADR-0022: English public documentation with one source per page

- Status: Accepted; migration is incremental
- Date: 2026-11-15
- Supersedes: the language requirement in decision 2 of
  [ADR-0008](0008-collaboration-conventions.md); its other decisions remain in force
- Revisit when: external contribution needs or language-specific user requirements change

## Context

External developers need to contribute algorithms and core features without reading Chinese.
Researchers need an English route through installation, PU semantics, and public Survey
reproduction. The maintenance team continues to lead core architecture decisions. Maintaining
parallel translations would add synchronization work without changing the software contracts.

## Decision

1. Maintain public documentation in English at its existing paths: `README.md`,
   `CONTRIBUTING.md`, the PR template, `docs/README.md`, all `docs/user/` Markdown,
   and all `docs/research/pu_survey/` Markdown.
2. Maintain five external developer guides in English: `architecture.md`,
   `project_structure.md`, `compatibility.md`, `new_algorithm_template.md`, and
   `comment_governance.md` under `docs/dev/`. State mandatory contribution rules in
   these guides and `CONTRIBUTING.md`; Chinese internal records are optional background
   for ordinary contributions.
3. Keep internal plans, other research notes, and historical ADRs in their existing
   language. Label Chinese references in public English navigation. Preserve the
   historical text of ADR-0008 with a supersession note.
4. Rewrite each public page in place and maintain one source. Remove the root Chinese
   README and its navigation references. Do not introduce language directories, parallel
   translations, translation metadata, or a translation platform/CI pipeline.
5. Keep code comments and docstrings in English. Runtime UI localization is separate;
   valid localized strings and test data remain allowed. The skill's language-specific
   `references/interpret.zh-CN.md` resource remains Chinese and keeps its current role.
6. Preserve API contracts, formulas, sampling/validation roles, commands, configuration
   identities, data rights qualifications, and historical artifact interpretation. Verify
   time-sensitive claims against evidence. Delete confirmed obsolete process notes
   instead of replacing them with new progress announcements.
7. Reuse existing quality gates. Check changed heading fragments separately because the
   local link gate does not validate fragments. Check skill compatibility using its
   existing synchronization gate and review referenced public guides for consistency.
8. Migrate contributor entry points, user concepts, how-to guides, API reference, and
   Survey reproduction in separate reviewable batches. Navigation must describe actual
   language availability during migration. Do not announce full completion early.

## Alternatives

- Parallel English and Chinese pages: rejected because keeping two technical contracts
  synchronized adds ongoing maintenance.
- Translate all research notes and historical decisions: rejected because their audience
  and maintenance cost differ from public usage and contribution documentation.
- Change APIs, reorganize document paths, or localize the UI during migration: separate
  concerns that would broaden review and compatibility risk.

## Consequences

International readers get a single maintained public contract. Internal Chinese history and
localized agent explanations remain available. New English headings require checking inbound
fragments, and technical review must distinguish implementation capability, executed training,
and verified numerical reproduction. API reference splitting under ADR-0017 is deferred to a
separate decision; this migration preserves its existing path.
