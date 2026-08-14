"""Rodent NAFLD inflammation scoring (Liang et al. 2014).

The dataset is mouse liver, so foci are scored with the rodent standard:
a focus is a cluster of >=5 inflammatory cells, counted per **3.1 mm2 field**,
and the field count maps to an ordinal grade 0-3.

  grade 0 normal   : < 0.5 foci/field
  grade 1 slight   : 0.5 - 1.0
  grade 2 moderate : 1.0 - 2.0
  grade 3 severe   : > 2.0
"""
from __future__ import annotations

FIELD_MM2 = 3.1  # Liang field-of-view area

_GRADE_LABELS = {0: "normal", 1: "slight", 2: "moderate", 3: "severe"}


def foci_per_field(fd_per_mm2: float, field_mm2: float = FIELD_MM2) -> float:
    """Convert focal density (foci/mm2) to foci per Liang field (3.1 mm2)."""
    return fd_per_mm2 * field_mm2


def grade(foci_per_field_value: float) -> int:
    """Liang inflammation grade 0-3 from foci-per-field."""
    if foci_per_field_value < 0.5:
        return 0
    if foci_per_field_value < 1.0:
        return 1
    if foci_per_field_value <= 2.0:
        return 2
    return 3


def grade_label(g: int) -> str:
    return _GRADE_LABELS.get(g, "?")
