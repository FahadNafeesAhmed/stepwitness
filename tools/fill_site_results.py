"""Write the measured end-to-end numbers into the page from the score files, never by hand.

Reads ``psr_results.json`` (scripts/evaluate_psr.py) and ``flag_results.json`` (scripts/evaluate_flags.py)
and rewrites three things in ``site/index.html``: the Stepwitness row of the results table, the
flag line under it, and the phase 3 and 4 rows of the status table.

Usage:
    python tools/fill_site_results.py --run runs/phase4/monitor_v3 [--site site/index.html]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_one(s: str, pattern: str, new: str) -> str:
    m = list(re.finditer(pattern, s, flags=re.S))
    assert len(m) == 1, f"expected exactly one match for {pattern!r}, found {len(m)}"
    return s[: m[0].start()] + new + s[m[0].end() :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="the run reported as the system (procedure guards on)")
    ap.add_argument("--run-plain", type=Path, default=None, help="the same verifier output replayed with B3 constants and no guards")
    ap.add_argument("--site", type=Path, default=ROOT / "site" / "index.html")
    args = ap.parse_args()

    psr = json.loads((args.run / "psr_results.json").read_text())
    flags = json.loads((args.run / "flag_results.json").read_text())
    a, e = psr["all"], psr["with_errors"]
    s = args.site.read_text(encoding="utf-8")

    def cells(g_all, g_err):
        return (f'<td class="num">{g_all["pos"]:.3f}</td><td class="num">{g_all["f1"]:.3f}</td><td class="num">{g_all["delay_s"]:.1f} s</td>'
                f'<td class="num">{g_err["pos"]:.3f}</td><td class="num">{g_err["f1"]:.3f}</td>')

    row = '<tr id="ours-row"><td><strong>Stepwitness</strong>, procedure guards on</td><td>renders only</td>' + cells(a, e) + '</tr>'
    s = replace_one(s, r'<tr id="ours-row">.*?</tr>', row)
    if args.run_plain is not None:
        pp = json.loads((args.run_plain / "psr_results.json").read_text())
        row_plain = '<tr id="ours-row-plain"><td>Stepwitness, guards off, B3 constants</td><td>renders only</td>' + cells(pp["all"], pp["with_errors"]) + '</tr>'
        s = replace_one(s, r'<tr id="ours-row-plain">.*?</tr>', row_plain)
        print(row_plain)

    ee, om = flags["execution_error_flags"], flags["end_omission_flags"]
    line = (
        '<p id="flags-line">Flags, which the baseline does not produce at all: execution-error flags matched a labelled wrong installation '
        f'within {flags["window_frames"] / 10:.0f} seconds with precision {ee["precision"]:.2f} and recall {ee["recall"]:.2f} '
        f'({ee["tp"]} right, {ee["fp"]} false alarms, {ee["fn"]} missed); end-of-recording omission flags had precision {om["precision"]:.2f} '
        f'and recall {om["recall"]:.2f}. Measured on all {a["n"]} test recordings, {e["n"]} of them with errors.</p>'
    )
    if 'id="flags-line"' in s:
        s = replace_one(s, r'<p id="flags-line">.*?</p>', line)
    else:
        anchor = '<p class="muted">Results are reported as measured, including the ones that fall short, with the frames that broke it.</p>'
        assert s.count(anchor) == 1
        s = s.replace(anchor, line + "\n" + anchor)

    s = replace_one(
        s,
        r"<tr><td>3</td><td>verifier and localizer trained on renders</td><td[^>]*>.*?</td></tr>",
        '<tr><td>3</td><td>verifier and localizer trained on renders</td><td class="ok">done, verifier v3 and detector, renders only</td></tr>',
    )
    s = replace_one(
        s,
        r"<tr><td>4</td><td>monitor over the 32 test recordings, scored with the authors' code</td><td[^>]*>.*?</td></tr>",
        f'<tr><td>4</td><td>monitor over the 32 test recordings, scored with the authors\' code</td><td class="ok">done, POS {a["pos"]:.3f}, F1 {a["f1"]:.3f}, delay {a["delay_s"]:.1f} s</td></tr>',
    )
    args.site.write_text(s, encoding="utf-8", newline="\n")
    print(row)
    print(line)


if __name__ == "__main__":
    main()
