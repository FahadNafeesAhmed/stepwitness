"""Derive the reviewable timeline from recognizer events: durations, cycle time, idle spans.

Everything here is arithmetic over the event list, kept separate from recognition so it can be
unit-tested without a model and reused by the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass

FPS = 10.0


@dataclass
class Span:
    start_frame: int
    end_frame: int

    @property
    def seconds(self) -> float:
        return (self.end_frame - self.start_frame) / FPS


TRACE_FIELDS = ("frame", "box", "top", "top_p", "err_component", "err_p")


def frame_evidence(events_dir, name: str, events: list[dict]) -> dict[int, dict]:
    """Per-frame evidence for drawing: the localizer box, the verifier's top state and confidence,
    and the most suspect component. Read from ``<name>.trace.json`` (one row per processed frame,
    written by the monitor); a processed frame where nothing was found is present with ``box`` None,
    so a drawing loop can clear the box. Older runs without a trace fall back to the evidence carried
    on events, which only exists at event frames."""
    import json
    from pathlib import Path

    trace = Path(events_dir) / f"{name}.trace.json"
    out: dict[int, dict] = {}
    if trace.exists():
        for row in json.loads(trace.read_text()):
            rec = dict(zip(TRACE_FIELDS, row))
            out[int(rec["frame"])] = rec
        return out
    for e in events:
        ev = e.get("evidence") or {}
        if ev.get("box"):
            out[int(e["frame"])] = {"frame": e["frame"], "box": ev["box"], "top": ev.get("top"), "top_p": ev.get("top_p"),
                                    "err_component": ev.get("err_component"), "err_p": ev.get("err_p")}
    return out


def completions(events: list[dict]) -> list[dict]:
    return sorted((e for e in events if e["type"] == "completion"), key=lambda e: e["frame"])


def flags(events: list[dict]) -> list[dict]:
    return [e for e in events if e["type"] in ("omission", "execution_error", "wrong_order")]


def step_durations(events: list[dict], first_frame: int = 0) -> list[dict]:
    """Time from the previous completion (or the start) to each completion."""
    out = []
    prev = first_frame
    for e in completions(events):
        out.append({"step": e["step"], "line": e.get("line"), "frame": e["frame"], "seconds": (e["frame"] - prev) / FPS})
        prev = e["frame"]
    return out


def cycle_time(events: list[dict], first_frame: int = 0) -> float | None:
    done = completions(events)
    if not done:
        return None
    return (done[-1]["frame"] - first_frame) / FPS


def idle_spans(events: list[dict], first_frame: int, last_frame: int, min_seconds: float = 20.0) -> list[Span]:
    """Stretches with no completion longer than ``min_seconds``: waiting, interruption, or a long step."""
    marks = [first_frame] + [e["frame"] for e in completions(events)] + [last_frame]
    spans = []
    for a, b in zip(marks, marks[1:]):
        if (b - a) / FPS >= min_seconds:
            spans.append(Span(a, b))
    return spans


def summary(events: list[dict], first_frame: int, last_frame: int) -> dict:
    done = completions(events)
    fl = flags(events)
    return {
        "completions": len(done),
        "flags": len(fl),
        "flags_by_type": {t: sum(1 for e in fl if e["type"] == t) for t in ("omission", "execution_error", "wrong_order")},
        "cycle_time_s": cycle_time(events, first_frame),
        "recording_s": (last_frame - first_frame) / FPS,
        "idle_spans": [{"start": s.start_frame, "end": s.end_frame, "seconds": s.seconds} for s in idle_spans(events, first_frame, last_frame)],
        "durations": step_durations(events, first_frame),
    }
