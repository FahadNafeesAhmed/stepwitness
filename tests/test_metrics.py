import random

import numpy as np
import pytest

from stepwitness.metrics import PAPER_COSTS, damerau_levenshtein, procedure_order_similarity


def dist(a, b):
    return damerau_levenshtein(list(a), list(b), **PAPER_COSTS)


def test_paper_examples():
    # Section 3.2.1: 'ACB' is penalised less than 'CAB' against ground truth 'ABC'.
    assert dist("ABC", "ACB") == 1.0
    assert dist("ABC", "CAB") == 2.0
    assert procedure_order_similarity([0, 1, 2], [0, 2, 1])[0] == pytest.approx(1 - 1 / 3)
    assert procedure_order_similarity([0, 1, 2], [2, 0, 1])[0] == pytest.approx(1 - 2 / 3)


def test_substitution_never_cheaper_than_delete_plus_insert():
    assert dist("ABC", "ABD") == 2.0
    assert dist("A", "B") == 2.0


def test_missing_and_extra_steps():
    assert dist("ABC", "AB") == 1.0  # one false negative
    assert dist("ABC", "ABCD") == 1.0  # one extra prediction
    assert dist("ABC", "") == 3.0
    assert dist("", "ABC") == 3.0
    assert dist("", "") == 0.0


def test_pos_is_capped_at_zero_and_perfect_at_one():
    assert procedure_order_similarity([0, 1, 2], [0, 1, 2]) == (1.0, 0.0)
    assert procedure_order_similarity([0, 1, 2], [3, 4, 5, 6, 7, 8])[0] == 0.0
    assert procedure_order_similarity([], [])[0] == 1.0


def test_unrestricted_transposition_with_symbols_between():
    # a = CA, b = ABC: unrestricted DL may transpose C and A with B inserted between them: 1 + 1 = 2.
    # The restricted variant (optimal string alignment) would need 3.
    assert dist("CA", "ABC") == 2.0


def test_repeated_symbols():
    assert dist("AAB", "ABA") == 1.0
    assert dist("ABAB", "BABA") == 2.0


@pytest.mark.parametrize("seed", range(5))
def test_matches_independent_unit_cost_implementation(seed):
    pytest.importorskip("rapidfuzz")
    from rapidfuzz.distance import DamerauLevenshtein

    rng = random.Random(seed)
    for _ in range(400):
        a = "".join(rng.choice("ABCDE") for _ in range(rng.randint(0, 9)))
        b = "".join(rng.choice("ABCDE") for _ in range(rng.randint(0, 9)))
        assert damerau_levenshtein(list(a), list(b)) == DamerauLevenshtein.distance(a, b), (a, b)


def test_compat_shim_matches_metrics_module():
    from stepwitness.compat.weighted_levenshtein import dam_lev

    ins = np.ones(128)
    dele = np.ones(128)
    sub = np.ones((128, 128)) * 2
    trans = np.ones((128, 128))
    for a, b in [("ABC", "ACB"), ("ABC", "CAB"), ("ABC", "ABD"), ("", "AB"), ("CA", "ABC")]:
        got = dam_lev(a, b, insert_costs=ins, delete_costs=dele, substitute_costs=sub, transpose_costs=trans)
        assert got == dist(a, b)
