import json

from stepwitness.timeline import frame_evidence


def test_frame_evidence_reads_trace_rows(tmp_path):
    rows = [[0, None, None, None, None, None], [1, [10.0, 20.0, 300.0, 200.0], "1,1,0,0,0,0,0,0,0,0,0", 0.81, 3, 0.07],
            [2, [11.0, 21.0, 300.0, 200.0], "1,1,1,0,0,0,0,0,0,0,0", 0.55, 2, 0.62]]
    (tmp_path / "rec.trace.json").write_text(json.dumps(rows))
    ev = frame_evidence(tmp_path, "rec", events=[])
    assert set(ev) == {0, 1, 2}  # a processed frame without a box is kept, with box None
    assert ev[0]["box"] is None
    assert ev[2]["err_component"] == 2 and ev[2]["err_p"] == 0.62
    assert ev[1]["top_p"] == 0.81


def test_frame_evidence_falls_back_to_events(tmp_path):
    events = [{"type": "completion", "frame": 40, "evidence": {"box": [1, 2, 3, 4], "top": "x", "top_p": 0.9}},
              {"type": "hold", "frame": 50, "evidence": {"box": None}}]
    ev = frame_evidence(tmp_path, "rec", events)
    assert list(ev) == [40]
    assert ev[40]["box"] == [1, 2, 3, 4] and ev[40]["err_p"] is None
