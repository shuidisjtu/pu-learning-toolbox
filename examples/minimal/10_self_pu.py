# ruff: noqa: N806

"""Train Self-PU with a separate clean validation set."""
# API reference: docs/user/reference/api/classifiers.md#selfpuclassifier-registered-as-self_pu

from sklearn.model_selection import train_test_split

from pu_toolbox.core.random import check_random_state
from pu_toolbox.estimators.deep import SelfPUClassifier
from pu_toolbox.preprocessing import GaussianSourceConfig, PULabelingConfig, simulate_pu_dataset
from pu_toolbox.prior import population_to_native_unlabeled


def main() -> None:
    rng = check_random_state(42)
    result = simulate_pu_dataset(
        GaussianSourceConfig(800, 8, 0.3, 2.0, rng),
        PULabelingConfig(
            "scar",
            "bernoulli",
            rng,
            c=0.4,
            ensure_labeled=True,
        ),
    )
    X, y_pu = result.observations.features, result.observations.labels
    y_true = result.audit.true_labels
    X_train, X_val, y_train, _, _, y_val = train_test_split(
        X,
        y_pu,
        y_true,
        test_size=0.2,
        stratify=y_true,
        random_state=42,
    )

    # Only the training partition determines the native-U prior conversion.
    # Clean validation provides extra supervision for meta/teacher selection, not test evidence.
    native_prior = population_to_native_unlabeled(
        0.3, float((y_train == 1).mean()), input_sampling="os"
    )
    classifier = SelfPUClassifier(
        risk_prior=native_prior,
        hidden_dim=32,
        warmup_epochs=1,
        self_paced_start=1,
        self_paced_end=3,
        distill_start=3,
        max_epochs=5,
        batch_size=128,
        require_validation=True,
        random_state=42,
        device="cpu",
    ).fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
    )

    print(classifier.get_pu_metadata())
    print(classifier.predict_proba(X_val[:5]))


if __name__ == "__main__":
    main()
