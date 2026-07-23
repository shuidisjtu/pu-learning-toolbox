"""Data preprocessing utilities for PU Learning Toolbox.

This subpackage provides reusable utilities for:

* PU / PNU data generation from fully-labeled data
  (:mod:`pu_toolbox.preprocessing.pu_labeling`)
* Data profiling / summary statistics
  (:mod:`pu_toolbox.preprocessing.profiling`)
"""

from pu_toolbox.preprocessing.profiling import (
    pnu_data_summary as pnu_data_summary,
)
from pu_toolbox.preprocessing.profiling import (
    pu_data_summary as pu_data_summary,
)
from pu_toolbox.preprocessing.profiling import (
    scar_diagnostic as scar_diagnostic,
)
from pu_toolbox.preprocessing.pu_labeling import (
    make_case_control_labels as make_case_control_labels,
)
from pu_toolbox.preprocessing.pu_labeling import (
    make_gaussian_pu_data as make_gaussian_pu_data,
)
from pu_toolbox.preprocessing.pu_labeling import (
    make_pnu_labels as make_pnu_labels,
)
from pu_toolbox.preprocessing.pu_labeling import (
    make_pu_labels as make_pu_labels,
)
from pu_toolbox.preprocessing.pu_labeling import (
    make_scar_dataset as make_scar_dataset,
)
from pu_toolbox.preprocessing.pu_labeling import (
    make_scar_labels as make_scar_labels,
)

__all__ = [
    "make_case_control_labels",
    "make_gaussian_pu_data",
    "make_pnu_labels",
    "make_pu_labels",
    "make_scar_dataset",
    "make_scar_labels",
    "pnu_data_summary",
    "pu_data_summary",
    "scar_diagnostic",
]
