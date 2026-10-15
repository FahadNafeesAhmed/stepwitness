"""Score monitor outputs with the IndustReal metric code, exactly as the baseline is scored.

Reads ``<name>.yhat.json`` files written by scripts/run_monitor.py, the ground-truth labels under the
recordings folder, and prints the same table as scripts/reproduce_baseline.py so the two line up.

Usage:
    python scripts/evaluate_psr.py --predictions runs/phase4/monitor_v2 --out runs/phase4/monitor_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stepwitness" / "compat"))
sys.path.insert(0, str(ROOT / "third_party" / "IndustReal" / "PSR"))
import psr_utils  # noqa: E402


def summarize(m: dict) -> dict:
    n = len(m["pos"])
    return {
        "n": n,
        "pos": psr_utils.mean_list(m["pos"]) if n else None,
        "f1": psr_utils.mean_list(m["f1"]) if n else None,
        "delay_s": psr_utils.mean_list(m["avg_delay"]) / psr_utils.FPS if n else None,
        "TP": m["system_TPs"],
        "FP": m["system_FPs"],
        "FN": m["system_FNs"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, default=ROOT / "data" / "industreal" / "recordings" / "test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="stepwitness")
    args = ap.parse_args()

    proc_info = psr_utils.get_procedure_info(ROOT / "third_party" / "IndustReal" / "PSR" / "procedure_info.json")
    m_all, m_err, m_ok = (psr_utils.initiate_metrics() for _ in range(3))
    per = {}
    for yfile in sorted(args.predictions.glob("*.yhat.json")):
        name = yfile.name.replace(".yhat.json", "")
        rec = args.recordings / name
        if not (rec / "PSR_labels.csv").exists():
            print(f"no labels for {name}, skipped")
            continue
        yhat = json.loads(yfile.read_text())
        gt = psr_utils.load_psr_labels(rec / "PSR_labels.csv")
        metrics = psr_utils.determine_performance(gt, yhat, proc_info, verbose=False)
        psr_utils.update_metrics(m_all, metrics)
        has_err = psr_utils.video_contains_errors(gt, proc_info, rec)
        psr_utils.update_metrics(m_err if has_err else m_ok, metrics)
        per[name] = {"has_errors": has_err, "pos": metrics["pos"], "f1": metrics["f1"], "avg_delay_frames": float(metrics["avg_delay"]),
                     "TP": metrics["system_TPs"], "FP": metrics["system_FPs"], "FN": metrics["system_FNs"], "n_pred": len(yhat), "n_gt": len(gt)}
        print(f"{name:14s} err={int(has_err)} POS={metrics['pos']:.3f} F1={metrics['f1']:.3f} delay={metrics['avg_delay'] / 10:.1f}s TP={metrics['system_TPs']} FP={metrics['system_FPs']} FN={metrics['system_FNs']} pred={len(yhat)} gt={len(gt)}")

    res = {"all": summarize(m_all), "with_errors": summarize(m_err), "without_errors": summarize(m_ok), "recordings": per}
    a, e = res["all"], res["with_errors"]
    fmt = lambda g, k, nd: (f"{g[k]:.{nd}f}" if g["n"] and g[k] is not None else "-")  # noqa: E731
    row = (f"| {args.label} | renders only | {fmt(a, 'pos', 3)} | {fmt(a, 'f1', 3)} | {fmt(a, 'delay_s', 1)} | "
           f"{fmt(e, 'pos', 3)} | {fmt(e, 'f1', 3)} | {fmt(e, 'delay_s', 1)} | {a['n']} / {e['n']} |") if a["n"] else "| no recordings scored |"
    print(row)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "psr_results.json").write_text(json.dumps(res, indent=1))
    (args.out / "psr_table_row.md").write_text(row + "\n")


if __name__ == "__main__":
    main()
