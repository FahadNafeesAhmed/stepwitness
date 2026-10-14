import pytest

from stepwitness.procedure import Procedure, labelled_states_industreal, parse_state, state_str

L = labelled_states_industreal()


def states(*numbers):
    return {L[n] for n in numbers}


def test_state_round_trip():
    assert state_str(parse_state("11100000000")) == "11100000000"
    assert parse_state("1,1,1,-1,0,0,0,0,0,0,0")[3] == -1
    with pytest.raises(ValueError):
        parse_state("111")


def test_assembly_reaches_exactly_its_labelled_states():
    p = Procedure.builtin("industreal_assembly")
    report = p.validate(states(*range(1, 14)))
    assert report.missing == set(), {state_str(s) for s in report.missing}
    assert report.extra == set(), {state_str(s) for s in report.extra}
    # chassis-without-pin and bracket-without-screw are reachable but unobservable
    assert parse_state("11000000000") in report.reachable_unobservable
    assert parse_state("11110111000") in report.reachable_unobservable


def test_maintenance_reaches_exactly_its_labelled_states():
    p = Procedure.builtin("industreal_maintenance")
    report = p.validate(states(13, 14, 15, 16, 17, 10, 18, 19, 20, 21, 22))
    assert report.missing == set(), {state_str(s) for s in report.missing}
    assert report.extra == set(), {state_str(s) for s in report.extra}


def test_both_procedures_cover_all_22_labelled_states():
    a = Procedure.builtin("industreal_assembly").validate(set()).reachable_observable
    m = Procedure.builtin("industreal_maintenance").validate(set()).reachable_observable
    assert a | m == set(L.values())


def test_assembly_allowed_sets_follow_the_partial_order():
    p = Procedure.builtin("industreal_assembly")
    assert p.allowed(set()) == {3, 9}
    assert p.allowed({3}) == {6, 9}
    assert p.allowed({9}) == {3, 15, 18}
    assert p.allowed({3, 6, 9, 15, 18}) == {21, 30}
    assert p.allowed({3, 6, 9, 15, 18, 21}) == {24, 30}
    assert p.allowed({3, 6, 9, 15, 18, 21, 24}) == {27, 30}
    assert p.allowed({3, 6, 9, 15, 18, 21, 24, 27, 30}) == set()


def test_maintenance_allowed_sets_follow_the_partial_order():
    p = Procedure.builtin("industreal_maintenance")
    assert p.allowed(set()) == {17, 20, 32}
    assert p.allowed({17, 20}) == {32}
    assert p.allowed({17, 20, 32}) == {11}
    assert p.allowed({17, 20, 32, 11}) == {12}
    assert p.allowed({17, 20, 32, 11, 12}) == {15, 18}
    assert p.allowed({17, 20, 32, 11, 12, 15, 18}) == {30}


def test_remove_then_install_same_component_ends_installed():
    p = Procedure.builtin("industreal_maintenance")
    done = {17, 20, 32, 11, 12, 15, 18, 30}
    assert state_str(p.state_after(done)) == "11101111111"
    assert p.state_after(done) == L[22]


def test_downset_is_required_for_state_after():
    p = Procedure.builtin("industreal_assembly")
    with pytest.raises(ValueError):
        p.state_after({6})  # pin without its chassis


def test_missing_steps_detects_what_the_state_implies():
    p = Procedure.builtin("industreal_assembly")
    observed = parse_state("11110110000")  # both subgoals complete
    assert p.missing_steps(observed, {3, 6}) == {9, 15, 18}
    assert p.missing_steps(observed, {3, 6, 9, 15, 18}) == set()


def test_cycle_is_rejected():
    from stepwitness.procedure import Step

    a = Step(3, 1, "install", frozenset({6}), "x", 1, "")
    b = Step(6, 2, "install", frozenset({3}), "x", 2, "")
    with pytest.raises(ValueError):
        Procedure("bad", parse_state("10000000000"), [a, b])
