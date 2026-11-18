# New Algorithm Contribution Template

## 1. Required declarations

A new algorithm must:

1. Implement the API contract (`fit`, `predict`, `decision_function`, `get_params`,
   `set_params`). Explicitly declare `label_semantics` as `"pu"`, `"pnu"`, or `"pn"`
   for a supervised target. This describes the **meaning** of labels in the main `fit(X, y)`;
   validating the values `{0, 1}` alone cannot distinguish PU from PN labels.

   Subclass `BasePUClassifier` and implement `fit(X, y_pu, *, class_prior=None,
   sample_weight=None)`, `_predict(X)`, and `_decision_function(X)`. `fit` returns `self`.
   Use the inherited public prediction wrappers so calls before fitting raise `NotFittedError`.
   `predict` returns an integer array of shape `(n_samples,)` containing only `{0, 1}`;
   `decision_function` returns finite scores of shape `(n_samples,)`, with higher scores
   indicating the positive class. Preserve sklearn-compatible parameter discovery and cloning
   through `get_params` and `set_params`. Declare `sample_weight_support` consistently with
   whether non-`None` weights are used, ignored, or rejected with `NotImplementedError`.
2. Declare the four architecture-capability fields as class attributes.
   `BasePUClassifier` supplies tabular defaults; deep algorithms must declare them explicitly:

   | Field | Legal values | Meaning |
   |---|---|---|
   | `native_architectures` | Subset of `{"mlp", "cnn"}` | Native architecture paths; the empty set derives `is_tabular_only=True` |
   | `input_ndims` | Nonempty subset of `{2, 4}` | Supported input dimensionalities |
   | `encoder_parameter` | `None` or a constructor parameter name | Parameter receiving an injected encoder; declarative metadata. Pipeline injection follows the constructor signature, rather than reading this field |
   | `trains_encoder` | `bool` | Whether the injected encoder is trained end to end |

3. Add the `AlgorithmMetadata` and concrete implementation class path together in
   `pu_toolbox/registry/builtin_methods.py`. Only implemented algorithms that satisfy their
   declared interfaces and capabilities may be registered; do not register placeholders.
   Registration associates metadata and class in one `register_method(metadata, estimator_cls)`
   operation. `source_status` and `maturity` still describe provenance and maturity.
4. Declare the registry-only fields in the entry literal: `aliases`, `deprecated_aliases`,
   `paper`, `supports_sparse`, `supports_gpu`, `upstream_url`, `license`, and
   `training_cost`. If a method requires an additional support set with clean labels, set
   `requires_clean_support=True` and make `fit` fail immediately when that support set is absent.
5. Choose the source of each of these seven entry fields **individually**:
   `family`, `assumption`, `scenario`, `requires_class_prior`, `source_status`,
   `backend`, and `maturity`.

   If the estimator class declares the field, registration synchronizes it into the registry;
   do not repeat it in `AlgorithmMetadata(...)`, where it would be overwritten.
   If the class does not declare it, write it in the entry literal or the dataclass default
   will apply. A declaration inherited from a concrete estimator class also counts; defaults
   inherited from `BasePUClassifier` or `BasePriorEstimator` do not.

   Registering a class does not mean omitting every field. For example,
   `class_prior_estimation` declares none of these fields, so all seven belong in its
   literal. The same rule puts `requires_class_prior` in the `pusb` and `lbe` literals.

   `test_static_entries_do_not_redeclare_class_fields` in `tests/test_builtin_methods.py`
   enforces both cases. **Known boundary:** class-only fields must never appear in entry
   literals. These are the four architecture fields, `label_semantics`,
   `class_prior_target`, and `risk_class_prior_target`. The internal
   [single-source map](single_source_map.md)
   records background in Chinese; the rules above are sufficient for external contributors.
6. If the method consumes a class prior, declare its target distribution consistently with the
   default fit risk view. `class_prior_target` accepts `"population"`,
   `"native_unlabeled"`, or `"training_table"`. An independent `risk_class_prior`
   parameter uses `risk_class_prior_target`, accepting `None`, `"population"`, or
   `"native_unlabeled"`. Here `population` means the positive proportion in the target
   population, `native_unlabeled` means the proportion in the original input U pool, and
   `training_table` means the proportion in the concatenated P/U training table. Declare
   the distribution actually consumed by the algorithm, rather than assuming these proportions
   are equal. The Pipeline resolves a population prior and converts it for each training fold;
   direct estimator calls must supply the prior required by that estimator's risk view.
   Further explanations and API details are in [prior semantics](../user/concepts/class_prior_semantics.md)
   and the [API contract](../user/reference/api/classifiers.md#classifiers-and-estimators).

## 2. Existing automated gates

These shared checks already exist; a contributor does not need to recreate them:

- `tests/contract/test_capability_declarations.py`: declaration legality, registry
  synchronization, derivation of `is_tabular_only`, and constructor-signature consistency.
  Registered classifiers must explicitly declare `label_semantics` on the class itself;
  omission fails the contract test.
- `tests/contract/test_classifier_baseline.py`: API contract and baseline behavior.
- Rule 1 of `scripts/check_doc_links.py`: referenced repository paths must exist.

Before merging, add algorithm-specific tests for the public interface and declared capabilities,
including relevant edge cases, as well as satisfying the shared contracts. Tests are not run
during registration or lookup. Register a small estimator factory in
`tests/estimator_factories.py` so the baseline suite can construct the registered method.

## 3. Declaring CNN support

If `native_architectures` includes `"cnn"`, provide:

- A CNN training smoke test.
- Input/output shape tests; validate encoder output with `validate_encoder_features`.
- CV-fold isolation tests to prevent weights leaking between folds.
- A fixed-seed test.
- GPU execution tests using the registered `gpu` pytest marker. They skip automatically
  without CUDA; run `pytest -m gpu` in the acceptance environment. See
  `tests/unit/estimators/test_nnpu_gpu.py`.
- A save/load/predict round-trip test.
- Fail-fast tests for unsupported architectures using `PUPipeline` architecture validation.

## 4. Minimal declarations

Class attributes for a deep algorithm with native MLP and CNN paths:

```python
native_architectures = frozenset({"mlp", "cnn"})
input_ndims = frozenset({2, 4})
encoder_parameter = "encoder"
trains_encoder = True
label_semantics = "pu"
```

Probe validation inside `fit` when the encoder is not `None`:

```python
representation_dim = validate_encoder_features(
    probe.flatten(start_dim=1), encoder_param_name="encoder"
)
```

A traditional tabular algorithm may inherit the tabular architecture defaults, but a registered
classifier must still explicitly declare `label_semantics = "pu"`. A three-way PNU method
declares `"pnu"`; a supervised oracle declares `"pn"`.
