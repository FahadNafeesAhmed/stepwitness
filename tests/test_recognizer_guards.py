"""The two procedure-grounded guards: reach (only states the procedure can be in soon earn credit)
and stable_frames (the top state must hold before it earns credit)."""
import numpy as np

from stepwitness.procedure import Procedure, labelled_states_industreal
from stepwitness.recognizer import Recognizer, RecognizerConfig
from stepwitness.verifier import tuple_to_code

L = labelled_states_industreal()
NO_ERR = np.zeros(11)


def code(n: int) -> str:
    return tuple_to_code(L[n])


def make(classes, **kw):
    return Recognizer(Procedure.builtin("industreal_assembly"), classes, RecognizerConfig(threshold=3.0, decay=0.75, **kw))


def onehot(classes, name, p=1.0):
    v = np.zeros(len(classes))
    v[classes.index(name)] = p
    if p < 1.0:
        rest = [i for i in range(len(classes)) if classes[i] != name]
        v[rest] = (1 - p) / len(rest)
    return v


def feed_parts(rec, classes, name, frames, start=0, p=1.0):
    events = []
    for f in range(start, start + frames):
        events.extend(rec.update_parts(onehot(classes, name, p), classes, NO_ERR, f))
    return events


def completions(events):
    return sorted(e["step"] for e in events if e["type"] == "completion")


def test_reach_blocks_a_state_far_ahead_but_allows_the_next_transition():
    classes = [code(1), code(5), code(9)]
    # without the guard, a verifier that reads state 9 at the start completes seven steps at once
    rec = make(classes)
    assert len(completions(feed_parts(rec, classes, code(9), 8))) == 7
    # with reach 2, the same reading earns nothing: state 9 is seven steps away
    rec = make(classes, reach=2)
    assert completions(feed_parts(rec, classes, code(9), 20)) == []
    assert rec.completed == set()
    # the real next transition (chassis + pin, two steps) still completes
    assert completions(feed_parts(rec, classes, code(5), 8, start=20)) == [3, 6]
    # and from there, state 9 is five steps away: still blocked at reach 2
    assert completions(feed_parts(rec, classes, code(9), 20, start=28)) == []


def test_reach_mass_is_not_renormalised():
    classes = [code(1), code(5), code(9)]
    rec = make(classes, reach=2, min_posterior=0.30)
    # 0.8 on the unreachable state, 0.1 on each of the others: after masking, the top reachable
    # state carries 0.1, below min_posterior, so nothing accumulates
    post = np.array([0.1, 0.1, 0.8])
    for f in range(20):
        rec.update_parts(post, classes, NO_ERR, f)
    assert rec.completed == set()


def test_stable_frames_rejects_a_flickering_top_state():
    classes = [code(1), code(5), code(7)]
    rec = make(classes, stable_frames=3)
    events = []
    for f in range(30):  # alternate between two readings every frame
        name = code(5) if f % 2 == 0 else code(7)
        events.extend(rec.update_parts(onehot(classes, name), classes, NO_ERR, f))
    assert completions(events) == []
    # a reading that holds completes as before
    assert completions(feed_parts(rec, classes, code(5), 8, start=30)) == [3, 6]
