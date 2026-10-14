"""Phase 0: reproduce the IndustReal PSR baselines B1, B2, B3 from the authors' published predictions.

Faithful to third_party/IndustReal/PSR/psr_baseline.py with two substitutions:

* the number of frames per recording comes from data/industreal/frame_counts.json, counted from the
  zip member lists by tools/extract_labels.py, instead of globbing extracted jpgs;
* the ``weighted_levenshtein`` import is served by stepwitness.compat, a pure Python port that is
  tested against an independent implementation.

Everything else, including the metric code, is the authors' own, imported from third_party.

Usage:
    python scripts/reproduce_baseline.py --out runs/phase0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stepwitness" / "compat"))  # provides weighted_levenshtein
sys.path.insert(0, str(ROOT / "third_party" / "IndustReal" / "PSR"))  # provides psr_utils

import psr_utils  # noqa: E402  (reference code)

IMPLEMENTATIONS = {"naive": "B1", "confidence": "B2", "expected": "B3"}
PUBLISHED = (
    "Published (paper Table 4): B3 real-trained 0.797 / 0.883 / 22.4 s on all, 0.731 / 0.816 / 20.4 s with errors; "
    "B3 synthetic-only 0.597 / 0.734 / 49.5 s on all, 0.571 / 0.731 / 71.4 s with errors."
)


def perform_psr_from_counts(config: dict, rec_dir: Path, n_frames: int):
    """psr_utils.perform_psr with the frame count injected instead of globbed from jpgs."""
    name = rec_dir.name
    procedure = "assy" if "assy" in name else "main" if "main" in name else None
    asd_predictions = psr_utils.load_asd_predictions(config["ads_dir"], rec_dir, n_frames)
    impl = config["implementation"]
    if impl == "naive":
        psr = psr_utils.NaivePSR(config)
    elif impl == "confidence":
        psr = psr_utils.AccumulatedConfidencePSR(config)
    elif impl == "expected":
        psr = psr_utils.AccumulatedConfidencePSR(config, procedure)
    else:
        raise ValueError(impl)
    for frame_n in range(n_frames):
        psr.update(asd_predictions[frame_n][1], frame_n)
    return psr


def single_set_parent(asd_dir: Path) -> Path:
    """The authors' loader globs the sub-directories of the directory it is given and keeps the one
    whose name contains the split name. Two prediction sets both contain 'test', so each set gets a
    parent directory of its own holding only that set."""
    parent = asd_dir.parent / f"_only_{asd_dir.name}"
    parent.mkdir(exist_ok=True)
    target = parent / asd_dir.name
    if not target.exists():
        try:
            target.symlink_to(asd_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(asd_dir, target)
    return parent


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


def run(asd_parent: Path, rec_root: Path, counts: dict[str, int], proc_info: list, impl: str) -> dict:
    recordings = sorted(psr_utils.get_recording_list(rec_root, test=True))
    config = {
        "implementation": impl,
        "ads_dir": asd_parent,
        "proc_info": proc_info,
        "cum_conf_threshold": 8,
        "cum_decay": 0.75,
        "conf_threshold": 0.5,
    }
    m_all, m_err, m_ok = (psr_utils.initiate_metrics() for _ in range(3))
    per_recording = {}
    for rec in recordings:
        key = f"{rec.parent.name}/{rec.name}"
        n_frames = counts[key]
        result = perform_psr_from_counts(config, rec, n_frames)
        gt = psr_utils.load_psr_labels(rec / "PSR_labels.csv")
        metrics = psr_utils.determine_performance(gt, result.y_hat, proc_info, verbose=False)
        psr_utils.update_metrics(m_all, metrics)
        has_errors = psr_utils.video_contains_errors(gt, proc_info, rec)
        psr_utils.update_metrics(m_err if has_errors else m_ok, metrics)
        per_recording[key] = {
            "has_errors": has_errors,
            "pos": metrics["pos"],
            "f1": metrics["f1"],
            "avg_delay_frames": float(metrics["avg_delay"]),
            "system_FPs": metrics["system_FPs"],
            "system_FNs": metrics["system_FNs"],
            "system_TPs": metrics["system_TPs"],
            "n_frames": n_frames,
            "n_pred": len(result.y_hat),
            "pred": result.y_hat,
            "gt": gt,
        }
    return {
        "all": summarize(m_all),
        "with_errors": summarize(m_err),
        "without_errors": summarize(m_ok),
        "recordings": per_recording,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "industreal")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "phase0")
    args = ap.parse_args()

    counts = json.loads((args.data / "frame_counts.json").read_text())
    proc_info = psr_utils.get_procedure_info(ROOT / "third_party" / "IndustReal" / "PSR" / "procedure_info.json")
    rec_root = args.data / "recordings"
    asd_sets = {
        "real_trained": args.data / "asd_results" / "ASD_IndustRealplusSynthetic_test",
        "synthetic_only": args.data / "asd_results" / "ASD_SyntheticOnly_test",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    rows = []
    for set_name, asd_dir in asd_sets.items():
        parent = single_set_parent(asd_dir)
        for impl, label in IMPLEMENTATIONS.items():
            r = run(parent, rec_root, counts, proc_info, impl)
            results[f"{set_name}/{impl}"] = r
            a, e = r["all"], r["with_errors"]
            rows.append(
                f"| {label} ({impl}) | {set_name} | {a['pos']:.3f} | {a['f1']:.3f} | {a['delay_s']:.1f} | "
                f"{e['pos']:.3f} | {e['f1']:.3f} | {e['delay_s']:.1f} | {a['n']} / {e['n']} |"
            )
            print(rows[-1])

    table = [
        "# Phase 0: reproduced PSR baselines",
        "",
        "Authors' published ASD predictions, authors' PSR and metric code, frame counts read from the zip listings.",
        "",
        "| Baseline | Predictions | POS all | F1 all | Delay all (s) | POS errors | F1 errors | Delay errors (s) | n all / n errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
        PUBLISHED,
    ]
    (args.out / "baseline_table.md").write_text("\n".join(table) + "\n")
    (args.out / "baseline_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {args.out / 'baseline_table.md'}")


if __name__ == "__main__":
    main()
