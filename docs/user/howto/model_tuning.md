# Model selection after Pipeline retirement

There is no replacement single-table candidate evaluator. For existing PU data, use direct estimator calls and design validation explicitly; never select on test truth. See [migration](pipeline.md).

For controlled studies with clean labels in all four roles, [ExperimentRunner](custom_experiments.md) supports a parameter grid for one model and independent PA/OA selection. It is not a multi-model comparator.
