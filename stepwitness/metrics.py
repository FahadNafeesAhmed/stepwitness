"""Procedure step recognition metrics, ported from IndustReal's psr_utils so results stay comparable.

The paper scores a recording with three numbers:

* procedure order similarity (POS): 1 - min(DamLev(gt, pred) / len(gt), 1), where DamLev is the
  unrestricted Damerau-Levenshtein distance with insert 1, delete 1, substitute 2 and transpose 1.
  Substitution costing 2 means a wrong step is never cheaper than a delete plus an insert, and
  transposition costing 1 means swapping two adjacent steps is a mild error.
* F1 over step completions, where a completion predicted before the true completion counts as a
  false positive.
* average delay in frames between a true completion and its recognition.

The authors call a Cython package for the distance. This file is a pure Python port with the same
semantics, verified in tests/test_metrics.py against an independent implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

Symbol = Any


class _Uniform(dict):
    """A mapping that returns one value for any key; the paper uses uniform costs."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __missing__(self, key):
        return self.value


class _Uniform2(dict):
    """A two-level mapping, ``m[s][t]``, returning one value for any pair."""

    def __init__(self, value: float):
        super().__init__()
        self.row = _Uniform(value)

    def __missing__(self, key):
        return self.row


def damerau_levenshtein(
    a: Sequence[Symbol],
    b: Sequence[Symbol],
    insert_costs: Mapping | None = None,
    delete_costs: Mapping | None = None,
    substitute_costs: Mapping | None = None,
    transpose_costs: Mapping | None = None,
) -> float:
    """Unrestricted weighted Damerau-Levenshtein distance between two sequences.

    ``insert_costs[s]`` and ``delete_costs[s]`` give the cost of inserting or deleting symbol ``s``.
    ``substitute_costs[s][t]`` gives the cost of replacing ``s`` by ``t`` and
    ``transpose_costs[s][t]`` the cost of swapping adjacent ``s`` followed by ``t`` in ``a``.
    Unrestricted means the transposed pair may have symbols deleted from between them in ``a`` and
    inserted between them in ``b``, each paid at its own cost. This is the textbook algorithm with
    the ``da`` (last row per symbol) and ``db`` (last matching column in this row) bookkeeping.
    """
    ins = insert_costs if insert_costs is not None else _Uniform(1.0)
    dele = delete_costs if delete_costs is not None else _Uniform(1.0)
    sub = substitute_costs if substitute_costs is not None else _Uniform2(1.0)
    trans = transpose_costs if transpose_costs is not None else _Uniform2(1.0)

    n, m = len(a), len(b)
    inf = float("inf")
    # d[i + 1][j + 1] holds the distance between a[:i] and b[:j]; row 0 and column 0 are the
    # sentinels the textbook writes as d[-1][.] and d[.][-1].
    d = [[inf] * (m + 2) for _ in range(n + 2)]
    d[1][1] = 0.0
    for i in range(1, n + 1):
        d[i + 1][1] = d[i][1] + dele[a[i - 1]]
    for j in range(1, m + 1):
        d[1][j + 1] = d[1][j] + ins[b[j - 1]]

    last_row: dict = {}  # symbol -> last 1-based position in a where it occurred
    for i in range(1, n + 1):
        ai = a[i - 1]
        last_col = 0  # last 1-based position in b, in this row, where a[i] matched
        for j in range(1, m + 1):
            bj = b[j - 1]
            k = last_row.get(bj, 0)
            l = last_col
            if ai == bj:
                cost = 0.0
                last_col = j
            else:
                cost = sub[ai][bj]
            best = min(
                d[i][j] + cost,  # substitute, or match at no cost
                d[i + 1][j] + ins[bj],  # insert b[j]
                d[i][j + 1] + dele[ai],  # delete a[i]
            )
            if k > 0 and l > 0:
                t = d[k][l]  # distance before the transposed pair
                for x in range(k, i - 1):  # symbols of a strictly between the pair
                    t += dele[a[x]]
                t += trans[a[k - 1]][ai]
                for y in range(l, j - 1):  # symbols of b strictly between the pair
                    t += ins[b[y]]
                best = min(best, t)
            d[i + 1][j + 1] = best
        last_row[ai] = i
    return d[n + 1][m + 1]


PAPER_COSTS = dict(
    insert_costs=_Uniform(1.0),
    delete_costs=_Uniform(1.0),
    substitute_costs=_Uniform2(2.0),
    transpose_costs=_Uniform2(1.0),
)


def procedure_order_similarity(gt: Sequence[int], pred: Sequence[int]) -> tuple[float, float]:
    """POS as defined in the IndustReal paper, section 3.2. Returns (score, distance)."""
    if len(gt) == 0:
        return (1.0 if len(pred) == 0 else 0.0), float(len(pred))
    distance = damerau_levenshtein(list(gt), list(pred), **PAPER_COSTS)
    score = 1.0 - min(distance / len(gt), 1.0)
    return score, distance
