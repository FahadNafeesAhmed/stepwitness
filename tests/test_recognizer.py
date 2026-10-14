import numpy as np
import pytest

from stepwitness.procedure import Procedure, labelled_states_industreal
from stepwitness.recognizer import Recognizer, RecognizerConfig
from stepwitness.verifier import tuple_to_code

L = labelled_states_industreal()


def code(n: int) -> str:
    return tuple_to_code(L[n])


def make(classes, threshold=3.0, **kw):
    cfg = RecognizerConfig(threshold=threshold, decay=0.75, **kw)
    return Recognizer(Procedure.builtin("industreal_assembly"), classes, cfg)


def onehot(classes, name, p=1.0):
    v = np.zeros(len(classes))
    v[classes.index(name)] = p
    if p < 1.0:
        rest = [i for i in range(len(classes)) if classes[i] != name]
        v[rest] = (1 - p) / len(rest)
    return v


def feed(rec, posterior, frames, start=0):
    events = []
    for f in range(start, start + frames):
        events.extend(rec.update(posterior, f))
    return events


def test_completes_steps_in_a_valid_order():
    classes = [code(1), code(5), code(7), code(8)]
    rec = make(classes)
    assert feed(rec, onehot(classes, code(1)), 5) == []
    # front chassis + pin appear together (state 5): both complete, allowed by the graph
    ev = feed(rec, onehot(classes, code(5)), 6, start=5)
    assert [e["step"] for e in ev if e["type"] == "completion"] == [3, 6]
    assert rec.completed == {3, 6}
    # rear chassis + front-rear pin (state 7), then rear-rear pin (state 8)
    ev = feed(rec, onehot(classes, code(7)), 6, start=11)
    assert sorted(e["step"] for e in ev if e["type"] == "completion") == [9, 15]
    ev = feed(rec, onehot(classes, code(8)), 6, start=17)
    assert [e["step"] for e in ev if e["type"] == "completion"] == [18]
    assert rec.state == L[8]


def test_parts_appearing_together_complete_together():
    # state 9 (bracket + screw + rear subgoal) observed after state 5: the rear chassis and its pins
    # are visibly there, so they complete in the same transition rather than being flagged
    classes = [code(1), code(5), code(9)]
    rec = make(classes)
    feed(rec, onehot(classes, code(5)), 6)
    ev = feed(rec, onehot(classes, code(9)), 6, start=6)
    done = sorted(e["step"] for e in ev if e["type"] == "completion")
    assert done == [9, 15, 18, 21, 24]
    assert not [e for e in ev if e["type"] == "omission"]


def test_visibly_missing_prerequisite_raises_omission_not_completion():
    # bracket and screw present while the rear chassis is absent: physically odd, logically a skip
    odd = "1,1,1,0,0,0,0,1,1,0,0"
    classes = [code(1), code(5), odd]
    rec = make(classes, omission_frames=4)
    feed(rec, onehot(classes, code(5)), 6)
    ev = feed(rec, onehot(classes, odd), 8, start=6)
    assert all(e["step"] not in (21, 24) for e in ev if e["type"] == "completion")
    om = [e for e in ev if e["type"] == "omission"]
    assert om and any(set(e["implies"]) & {15, 18} for e in om)


def test_finalize_reports_steps_never_completed_as_omissions():
    classes = [code(1), code(5)]
    rec = make(classes)
    feed(rec, onehot(classes, code(5)), 6)
    ev = rec.finalize(frame=100)
    assert {e["type"] for e in ev} == {"omission"}
    assert sorted(e["step"] for e in ev) == [9, 15, 18, 21, 24, 27, 30]
    assert all(e["at_end"] for e in ev)


def test_execution_error_holds_the_step_open_and_flags():
    bad5 = "1,1,1,1,0,-1,0,0,0,0,0"  # front-rear pin wrong at state 7
    classes = [code(1), code(5), code(7), bad5]
    rec = make(classes, error_frames=4)
    feed(rec, onehot(classes, code(5)), 6)
    v = np.zeros(len(classes))
    v[classes.index(code(7))] = 0.3
    v[classes.index(bad5)] = 0.7
    ev = feed(rec, v, 10, start=6)
    assert any(e["type"] == "execution_error" and e["component"] == 5 for e in ev)
    assert 15 not in rec.completed  # the front-rear pin did not complete
    assert 9 in rec.completed  # the rear chassis itself, supported by both states, did complete


def test_hold_when_nothing_is_confident():
    classes = [code(1), code(5), code(7)]
    rec = make(classes, hold_frames=5, hold_floor=0.5)
    flat = np.full(len(classes), 1 / len(classes))
    ev = feed(rec, flat, 6)
    assert [e["type"] for e in ev] == ["hold"]


def test_y_hat_matches_paper_format():
    classes = [code(1), code(5)]
    rec = make(classes)
    feed(rec, onehot(classes, code(5)), 6)
    proc_info = [{"id": 3, "description": "Install front chassis"}, {"id": 6, "description": "Install front chassis pin"}]
    y = rec.y_hat(proc_info)
    assert y[0]["id"] == 3 and y[0]["description"] == "Install front chassis" and y[0]["conf"] == 1
    assert y[1]["id"] == 6
