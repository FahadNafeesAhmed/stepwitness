import numpy as np

from stepwitness.procedure import Procedure, labelled_states_industreal
from stepwitness.recognizer import Recognizer, RecognizerConfig
from stepwitness.verifier import tuple_to_code
from stepwitness.verifier_v3 import STATE_CODES, labels_from_codes

L = labelled_states_industreal()


def code(n):
    return tuple_to_code(L[n])


def onehot_state(name, p=1.0):
    v = np.zeros(len(STATE_CODES))
    v[STATE_CODES.index(name)] = p
    return v


def test_state_codes_are_the_22_labelled_states_in_order():
    assert len(STATE_CODES) == 22
    assert STATE_CODES[0] == code(1) and STATE_CODES[12] == code(13) and STATE_CODES[21] == code(22)


def test_labels_from_codes_split_state_and_error():
    ys, ye = labels_from_codes([code(7), "1,1,1,1,0,-1,0,0,0,0,0"])
    assert ys.tolist() == [6, 6]  # both are state 7 (index 6)
    assert ye[0].sum() == 0 and ye[1][5] == 1 and ye[1].sum() == 1


def test_update_parts_completes_and_flags():
    cfg = RecognizerConfig(threshold=3.0, error_frames=4)
    rec = Recognizer(Procedure.builtin("industreal_assembly"), STATE_CODES, cfg)
    no_err = np.zeros(11)
    for f in range(6):
        rec.update_parts(onehot_state(code(5)), STATE_CODES, no_err, f)
    assert rec.completed == {3, 6}
    # state 7 seen, but the front-rear pin (component 5) reads as wrong
    err = np.zeros(11)
    err[5] = 0.8
    ev = []
    for f in range(6, 18):
        ev.extend(rec.update_parts(onehot_state(code(7)), STATE_CODES, err, f))
    assert any(e["type"] == "execution_error" and e["component"] == 5 for e in ev)
    assert 9 in rec.completed and 15 not in rec.completed
