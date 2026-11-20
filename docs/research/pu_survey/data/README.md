# Survey Data Sources and Interpretation Limits

The loaders and current class mappings are defined by `SurveyDatasetSpec` and
`DatasetProvenance` in [`datasets.py`](../../../../pu_toolbox/experiment/datasets.py).
Each run's split manifest records its actual source, seed, partition indices, and train-only
preprocessing. Raw data and training results are not stored in the repository.

## Sources and licensing

| Dataset slot | Source and licensing interpretation |
|---|---|
| MNIST | [Original source](http://yann.lecun.com/exdb/mnist/); torchvision supplies download definitions. A mirror or package license does not automatically authorize use of the original data |
| Fashion-MNIST | [Publisher](https://github.com/zalandoresearch/fashion-mnist); the recorded publisher repository declares MIT |
| CIFAR-10 | [Publisher](https://www.cs.toronto.edu/~kriz/cifar.html); retain the original source without inferring a data license from a dependency's software license |
| IMDB | [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/); retain its research-use conditions and source statements, without inferring an additional license |
| 20 Newsgroups | [Original source](http://qwone.com/~jason/20Newsgroups/); scikit-learn's BSD software license does not authorize the news text. The loader explicitly maps 20 groups to 7 superclasses |
| Spambase / Connect-4 | [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase) / [UCI Connect-4](https://archive.ics.uci.edu/dataset/26/connect-4); the recorded pages state CC BY 4.0, requiring attribution. Connect-4's b/o/x values are categorical features |
| `adni` slot | The Alzheimer MRI substitute described below; do not describe it as approved ADNI data |

The collection entry point, `scripts/collect_survey_public_data.py`, uses recorded URLs and
digests and refuses to overwrite an existing mismatched file. MNIST and Fashion-MNIST now
share an IDX reader; their source configurations record the previously reviewed URLs and
checksums from torchvision without importing it. The 20 Newsgroups checksums come from
scikit-learn. These are checksums recorded by dependency projects, not source licenses.
No publisher digest is registered for UCI Connect-4, so its local
SHA-256 establishes byte identity only. Preserve licensing uncertainty from the source records:
an unknown license is not a grant. Dataset provenance uses `not_stated_by_source` when no
license is transcribed, retaining the date and notes of the source review.

## Alzheimer MRI substitute

The project's `adni` slot uses a personal mirror of Kaggle's
*Alzheimer's Dataset (4 class of Images)*:
[`Jahnabi19/AD-Dataset`](https://github.com/Jahnabi19/AD-Dataset/tree/d90729118954bbedfe481dea8a53b4f3ee5a21b2),
pinned to commit `d90729118954bbedfe481dea8a53b4f3ee5a21b2`.
This is not ADNI data and carries no ADNI data-use approval. The mirror states no license;
the original Kaggle page was recorded as License: Unknown at the time of review.

- There are 6400 images: NonDemented 3200, VeryMildDemented 2240, ModerateDemented 64,
  and MildDemented 896.
- NonDemented is positive; the other three classes are negative, giving population prior 0.5.
  This differs from PU-Bench's alternative binary mapping.
- The originals are grayscale JPEGs, resized by the loader to 128×128. Normalization
  statistics are fitted only on the actual train partition.
- The source has no subject IDs. Current splits use sample indices, so subject-level isolation
  cannot be verified and slices from the same subject may cross partitions. Paths requiring
  verified subject/group evidence reject missing evidence. The explicit clean-support route
  instead records `group_policy="unverifiable"`, `subject_isolation="not_checkable"`, the
  known risk, and an owner-approval identifier; that record does not establish subject isolation.

Archive digests, class counts, and identity facts are retained in
[`adni_substitute_receipt_20261012.json`](adni_substitute_receipt_20261012.json).
Its historical acceptance status describes the materials at that time and does not
automatically determine the validity of new results.

## Splits, support, and retained machine-readable facts

The four roles are train / pu_val / clean_val / test. True test labels are used only for final
evaluation. Independent clean support carved from train removes those rows from train and
records all role indices and the actual positive/negative label budget. Support and selection
validation cannot reuse samples. The project's configured support sizes are 100 for Spambase
and 500 for the other configured support datasets; actual execution must also follow the
support manifest. Support selection is label-blind, uniform without replacement, and uses
the predeclared split seed; it does not retry until a desired class composition appears.

The original PULNS paper covers MNIST, CIFAR-10, and Spambase; the other datasets are project
extensions. MNIST's even/odd and CIFAR's vehicle/animal mappings match the paper. Spambase
retains spam-positive labeling; the paper's original encoding has not been independently
confirmed. The paper's γ is the positive proportion in U, not the labeling frequency c.
Its support sizes must not silently become new-experiment defaults, and the authors' code
has not been confirmed.

Actual source, file digests, and split identity come from the provenance and split manifest
generated for each preparation. Early pilot split indices and receipt records have been
retired; historical material is available at the pre-cleanup commit `7d11f2c4`.
The P3MIX literature anchors retain paper sources, numbers, and comparability limits without
binding early pilot splits. Other method-source JSON and P3MIX/PULNS literature anchors retain
external source facts. [Method cards](../../method_cards/) provide optional Chinese background
on implementation differences; these records are not mandatory dependencies for training,
report generation, or human approval.
