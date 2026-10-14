"""Score the monitor's flags against the dataset's error annotations.

The paper's PSR metric scores completions only. This script scores the two things the paper does
not: execution-error flags against the "Incorrectly installed <component>" events in
``PSR_labels_with_errors.csv`` (ids 3k + 1), and end-of-recording omission flags against the
expected steps the ground truth never contains.

An execution-error flag on component k counts as a hit if the ground truth has an incorrect
installation of k whose frame lies within ``--window`` frames before the flag, or that has not been
undone by then (a "Remove k" entry comes later). Each ground-truth error can be hit once.

Usage:
    python scripts/evaluate_flags.py --events runs/phase4/monitor_v2 --out runs/phase4/monitor_v2 --window 300
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.procedure import Procedure  # noqa: E402


def load_with_errors(path: Path) -> list[dict]:
    rows = []
    with open(path) as fp:
        for r in csv.reader(fp):
            rows.append({"frame": int(r[0][:-4]), "id": int(r[1]), "description": r[2]})
    return rows


def gt_errors(rows: list[dict]) -> list[dict]:
    """Incorrect installations with the frame they were undone (if ever)."""
    out = []
    for r in rows:
        if r["id"] % 3 == 1:
            comp = r["id"] // 3
            undo = [q["frame"] for q in rows if q["frame"] > r["frame"] and q["id"] == 3 * comp + 2]
            out.append({"component": comp, "frame": r["frame"], "undone_at": min(undo) if undo else None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, default=ROOT / "data" / "industreal" / "recordings" / "test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=300, help="frames after a ground-truth error within which a flag counts")
    args = ap.parse_args()

    tp = fp = fn = 0
    om_tp = om_fp = om_fn = 0
    per = {}
    for ef in sorted(args.events.glob("*.events.json")):
        name = ef.name.replace(".events.json", "")
        rec = args.recordings / name
        if not (rec / "PSR_labels_with_errors.csv").exists():
            continue
        events = json.loads(ef.read_text())
        rows = load_with_errors(rec / "PSR_labels_with_errors.csv")
        errors = gt_errors(rows)
        flags = [e for e in events if e["type"] == "execution_error"]
        matched = set()
        hits = 0
        for f in flags:
            ok = False
            for i, g in enumerate(errors):
                if i in matched or g["component"] != f["component"]:
                    continue
                active_until = g["undone_at"] if g["undone_at"] is not None else 10**9
                if g["frame"] <= f["frame"] <= max(g["frame"] + args.window, active_until):
                    matched.add(i)
                    ok = True
                    break
            hits += ok
        r_tp, r_fp, r_fn = hits, len(flags) - hits, len(errors) - len(matched)
        tp, fp, fn = tp + r_tp, fp + r_fp, fn + r_fn

        # omissions at the end: expected steps of the procedure never completed in the ground truth
        proc = Procedure.builtin("industreal_assembly" if "assy" in name else "industreal_maintenance")
        gt_ids = {r["id"] for r in rows}
        expected_missing = {i for i in proc.steps if i not in gt_ids}
        flagged = {e["step"] for e in events if e["type"] == "omission" and e.get("at_end")}
        o_tp = len(flagged & expected_missing)
        o_fp = len(flagged - expected_missing)
        o_fn = len(expected_missing - flagged)
        om_tp, om_fp, om_fn = om_tp + o_tp, om_fp + o_fp, om_fn + o_fn
        per[name] = {"gt_errors": len(errors), "flags": len(flags), "hits": hits, "omissions_expected": sorted(expected_missing), "omissions_flagged": sorted(flagged)}
        print(f"{name:14s} gt_err={len(errors)} flags={len(flags)} hits={hits}  omissions exp={sorted(expected_missing)} flagged={sorted(flagged)}")

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}

    res = {"execution_error_flags": prf(tp, fp, fn), "end_omission_flags": prf(om_tp, om_fp, om_fn), "window_frames": args.window, "recordings": per}
    print(json.dumps({k: res[k] for k in ("execution_error_flags", "end_omission_flags")}, indent=1))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "flag_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
