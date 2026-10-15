"""The release rule: with reach on, a state the procedure cannot reach earns nothing until the verifier
has insisted on it for release_frames frames with no completion in between; then it is trusted."""
import numpy as np

from stepwitness.procedure import Procedure, labelled_states_industreal
from stepwitness.recognizer import Recognizer, RecognizerConfig
from stepwitness.verifier import tuple_to_code

L = labelled_states_industreal()
NO_ERR = np.zeros(11)


def code(n: int) -> str:
    return tuple_to_code(L[n])


def onehot(classes, name):
    v = np.zeros(len(classes))
    v[classes.index(name)] = 1.0
    return v


def test_release_trusts_a_sustained_unreachable_reading():
    classes = [code(1), code(5), code(9)]
    cfg = RecognizerConfig(threshold=3.0, decay=0.75, reach=2, release_frames=10)
    rec = Recognizer(Procedure.builtin("industreal_assembly"), classes, cfg)
    events = []
    for f in range(9):  # nine frames of a reading seven steps ahead: blocked
        events.extend(rec.update_parts(onehot(classes, code(9)), classes, NO_ERR, f))
    assert rec.completed == set()
    for f in range(9, 20):  # the tenth confident frame releases the guard; credit then accumulates
        events.extend(rec.update_parts(onehot(classes, code(9)), classes, NO_ERR, f))
    assert len([e for e in events if e["type"] == "completion"]) == 7
    assert rec.state == L[9]


def test_release_counter_resets_on_a_completion():
    classes = [code(1), code(5), code(9)]
    cfg = RecognizerConfig(threshold=3.0, decay=0.75, reach=2, release_frames=10)
    rec = Recognizer(Procedure.builtin("industreal_assembly"), classes, cfg)
    for f in range(8):  # eight stuck frames
        rec.update_parts(onehot(classes, code(9)), classes, NO_ERR, f)
    for f in range(8, 14):  # a reachable transition completes and resets the counter
        rec.update_parts(onehot(classes, code(5)), classes, NO_ERR, f)
    assert rec.completed == {3, 6}
    for f in range(14, 22):  # eight more stuck frames: still short of ten since the reset
        rec.update_parts(onehot(classes, code(9)), classes, NO_ERR, f)
    assert rec.completed == {3, 6}
