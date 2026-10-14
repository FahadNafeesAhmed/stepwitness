"""Map the loose shells of the complete states (13 and 22) to procedure components using set
differences between states whose only difference is one component.

Positions are keyed relative to the chassis base so files with different origins still match.
The result is a draft; tools/blender_highlight_shells.py renders every shell for a visual check and
the reviewed mapping is committed as stepwitness/procedures/industreal_shells.json.

Usage:
    python tools/map_components.py runs/phase2/state_parts.json --out runs/phase2/component_shells.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

Q = 3.0


def anchor(shells):
    """Centre of the longest silver shell: the 82 mm front axle that belongs to the base and is
    present, at the same place, in every state."""
    silvers = [s for s in shells if s["colour"] == "Metallic silver"] or shells
    return max(silvers, key=lambda s: max(s["dims"]))["center"]


def key(shell, origin):
    rel = [c - o for c, o in zip(shell["center"], origin)]
    return (shell["colour"], tuple(round(v / Q) for v in rel), tuple(round(v / Q) for v in shell["dims"]))


def keyed(states, name):
    shells = states[name]
    o = anchor(shells)
    return {key(s, o): i for i, s in enumerate(shells)}


def diff(states, a, b):
    """Indices of shells of state a that have no counterpart in state b."""
    ka, kb = keyed(states, a), keyed(states, b)
    return sorted(ka[k] for k in ka if k not in kb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    states = json.loads(args.parts.read_text())["states"]

    # each line: component -> shells of a complete state that disappear in a state lacking only that component
    s13 = {
        "rear-rear chassis pin (6)": diff(states, "state13", "state14"),
        "front-rear chassis pin (5)": diff(states, "state13", "state15"),
        "rear wheel assembly (10)": diff(states, "state13", "state10"),
        "front wheel assembly (9)": diff(states, "state13", "state12"),
        "front bracket + screw (7, 8) + front wheels (9)": diff(states, "state13", "state11"),
        "front chassis + pin (1, 2) + bracket + screw + front wheels": diff(states, "state13", "state4") if "state4" in states else [],
        "everything but base (sanity)": diff(states, "state13", "state1"),
    }
    s22 = {
        "rear wheel assembly (10)": diff(states, "state22", "state21"),
        "rear-rear chassis pin (6)": diff(states, "state22", "state19"),
        "front-rear chassis pin (5)": diff(states, "state22", "state20"),
        "short rear chassis + pins + wheels (4, 5, 6, 10)": diff(states, "state22", "state18"),
    }
    report = {"state13": s13, "state22": s22}
    for st, groups in report.items():
        n = len(states[st])
        print(f"== {st} ({n} shells) ==")
        for name, idx in groups.items():
            print(f"  {name}: {idx}")
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
