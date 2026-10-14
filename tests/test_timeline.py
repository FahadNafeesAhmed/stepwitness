from stepwitness.timeline import cycle_time, idle_spans, step_durations, summary

EVENTS = [
    {"type": "completion", "step": 3, "line": 2, "frame": 100},
    {"type": "completion", "step": 6, "line": 3, "frame": 100},
    {"type": "execution_error", "step": 15, "component": 5, "frame": 400},
    {"type": "completion", "step": 9, "line": 4, "frame": 700},
    {"type": "omission", "step": 30, "component": 10, "frame": 1500, "at_end": True},
]


def test_durations_and_cycle_time():
    d = step_durations(EVENTS, first_frame=0)
    assert [x["step"] for x in d] == [3, 6, 9]
    assert d[0]["seconds"] == 10.0 and d[1]["seconds"] == 0.0 and d[2]["seconds"] == 60.0
    assert cycle_time(EVENTS, 0) == 70.0


def test_idle_spans_find_long_gaps():
    spans = idle_spans(EVENTS, first_frame=0, last_frame=1500, min_seconds=30)
    assert [(s.start_frame, s.end_frame) for s in spans] == [(100, 700), (700, 1500)]


def test_summary_counts_flags():
    s = summary(EVENTS, 0, 1500)
    assert s["completions"] == 3 and s["flags"] == 2
    assert s["flags_by_type"] == {"omission": 1, "execution_error": 1, "wrong_order": 0}
    assert s["recording_s"] == 150.0
