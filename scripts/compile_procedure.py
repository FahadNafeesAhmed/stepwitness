"""Phase 1: compile a numbered work instruction into a procedure graph with a local language model.

The model drafts; a person reviews. This script makes the draft, validates it against the dataset's
labelled states, and reports how it differs from the reviewed graph that is committed under
stepwitness/procedures. The model only has to read the sheet and decide, per step, which component
moves, whether it is installed or removed, and which earlier lines must be done first. Step ids are
computed here, never by the model, so a wrong id is impossible.

Usage:
    python scripts/compile_procedure.py --sheet docs/instructions/industreal_assembly.md \
        --name industreal_assembly --start-state 10000000000 --labelled 1-13

Needs an Ollama server on localhost (default model qwen3:8b).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stepwitness.procedure import (  # noqa: E402
    COMPONENTS,
    Procedure,
    install_id,
    labelled_states_industreal,
    remove_id,
)

OBSERVABLE_ONLY_WITH = {"1": [2], "3": [5, 6], "4": [5, 6], "7": [8]}

SYSTEM = """You convert a numbered work instruction for assembling a construction-toy car into a procedure graph.

Components, with their index k:
{components}

Return only JSON of this shape, nothing else:
{{"steps": [{{"line": <int>, "component": <k>, "kind": "install" | "remove", "requires_lines": [<int>, ...], "subgoal": "<short name>"}}, ...]}}

Rules:
- One entry per instruction line that performs an action on exactly one component. Skip lines that only describe a starting condition, such as "Start from the base" or "Start from the assembled car".
- "kind" is "install" when the component is put on or secured, "remove" when it is taken off or pulled out.
- "requires_lines" lists the line numbers of the actions that must already be done before this one, according to the sheet's own text about order. A pin that secures a part requires the line that fits that part. When the sheet says a step needs a subgoal complete, list the lines the sheet names for that subgoal, including the pin lines. Do not add an order the sheet does not state.
- "subgoal" is a short name for the group the sheet describes (for example front, rear, bracket, wheels, strip, refit).
"""


def ollama_chat(model: str, system: str, user: str, think: bool = False, url: str = "http://127.0.0.1:11434/api/chat") -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "format": "json",
        "stream": False,
        "think": think,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode())
    return body["message"]["content"]


def draft_to_procedure(name: str, start_state: str, draft: dict) -> Procedure:
    by_line = {int(s["line"]): s for s in draft["steps"]}
    steps = []
    for line, s in sorted(by_line.items()):
        k = int(s["component"])
        kind = s["kind"]
        sid = install_id(k) if kind == "install" else remove_id(k)
        req = []
        for rl in s.get("requires_lines", []):
            r = by_line.get(int(rl))
            if r is None:
                continue
            rk = int(r["component"])
            req.append(install_id(rk) if r["kind"] == "install" else remove_id(rk))
        steps.append(
            {
                "id": sid,
                "component": k,
                "kind": kind,
                "requires": sorted(set(req)),
                "subgoal": s.get("subgoal", ""),
                "line": line,
                "text": "",
            }
        )
    return Procedure.from_dict(
        {"name": name, "start_state": start_state, "observable_only_with": OBSERVABLE_ONLY_WITH, "steps": steps}
    )


def parse_labelled(spec: str) -> set:
    labelled = labelled_states_industreal()
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= {labelled[i] for i in range(int(a), int(b) + 1)}
        else:
            out.add(labelled[int(part)])
    return out


def compare(draft: Procedure, reviewed: Procedure) -> list[str]:
    lines = []
    d_ids, r_ids = set(draft.steps), set(reviewed.steps)
    for i in sorted(r_ids - d_ids):
        lines.append(f"missing step {i} ({reviewed.steps[i].kind} {COMPONENTS[reviewed.steps[i].component]})")
    for i in sorted(d_ids - r_ids):
        lines.append(f"extra step {i} ({draft.steps[i].kind} {COMPONENTS[draft.steps[i].component]})")
    for i in sorted(d_ids & r_ids):
        a, b = draft.steps[i].requires, reviewed.steps[i].requires
        if a != b:
            lines.append(f"step {i}: draft requires {sorted(a)}, reviewed requires {sorted(b)}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--start-state", required=True)
    ap.add_argument("--labelled", default="", help="labelled state numbers, e.g. 1-13 or 13,14,15")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--think", action="store_true", help="let the model reason before answering (slower)")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. attempt3")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "phase1")
    args = ap.parse_args()

    components = "\n".join(f"  k={k}: {name}" for k, name in enumerate(COMPONENTS))
    sheet = args.sheet.read_text()
    raw = ollama_chat(args.model, SYSTEM.format(components=components), sheet, think=args.think)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.tag:
        args.name = f"{args.name}.{args.tag}"
    (args.out / f"{args.name}.raw.json").write_text(raw)
    draft = json.loads(raw)
    proc = draft_to_procedure(args.name, args.start_state, draft)
    (args.out / f"{args.name}.compiled.json").write_text(json.dumps(proc.to_dict(), indent=2))

    base_name = args.name.split(".")[0]
    report = [f"# Compiled {args.name} with {args.model}{' (thinking)' if args.think else ''}", ""]
    if args.labelled:
        v = proc.validate(parse_labelled(args.labelled))
        report.append(f"validation against labelled states: {'OK' if v.ok else 'FAIL'}")
        report.append(f"  reachable observable: {len(v.reachable_observable)}, missing: {len(v.missing)}, extra: {len(v.extra)}")
    reviewed_path = ROOT / "stepwitness" / "procedures" / f"{base_name}.json"
    if reviewed_path.exists():
        diffs = compare(proc, Procedure.load(reviewed_path))
        report.append(f"differences from the reviewed graph: {len(diffs)}")
        report.extend(f"  - {d}" for d in diffs)
    text = "\n".join(report)
    print(text)
    (args.out / f"{args.name}.report.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
