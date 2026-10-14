"""Pure Python stand-in for the ``weighted_levenshtein`` Cython package used by IndustReal's psr_utils.

Only ``dam_lev`` is provided, with the same signature: two strings, per-character cost arrays indexed
by character code (128 entries, or 128 x 128 for substitute and transpose), distance out.
"""

from __future__ import annotations

import numpy as np

from stepwitness.metrics import damerau_levenshtein


class _ArrayCosts:
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, key):
        return float(self.arr[ord(key)])


class _RowCosts:
    def __init__(self, row):
        self.row = row

    def __getitem__(self, key):
        return float(self.row[ord(key)])


class _ArrayCosts2:
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, key):
        return _RowCosts(self.arr[ord(key)])


def dam_lev(
    str1: str,
    str2: str,
    insert_costs=None,
    delete_costs=None,
    substitute_costs=None,
    transpose_costs=None,
) -> float:
    insert_costs = np.ones(128) if insert_costs is None else insert_costs
    delete_costs = np.ones(128) if delete_costs is None else delete_costs
    substitute_costs = np.ones((128, 128)) if substitute_costs is None else substitute_costs
    transpose_costs = np.ones((128, 128)) if transpose_costs is None else transpose_costs
    return damerau_levenshtein(
        list(str1),
        list(str2),
        insert_costs=_ArrayCosts(insert_costs),
        delete_costs=_ArrayCosts(delete_costs),
        substitute_costs=_ArrayCosts2(substitute_costs),
        transpose_costs=_ArrayCosts2(transpose_costs),
    )
