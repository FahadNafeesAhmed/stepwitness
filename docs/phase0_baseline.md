# Phase 0: reproducing the published baseline

## Why first

Every number this project claims is relative to the IndustReal procedure step recognition baseline
(Schoonbeek et al., WACV 2024, Table 4). If that baseline does not reproduce on this machine, nothing
measured later means anything. So the first measurement is theirs, not ours.

## What was run

The authors published the predictions of their assembly-state detector on the test split, for the
real-trained model and the synthetic-only model, plus the PSR code and the metric code. Reproducing
the table therefore needs no GPU: it replays their predictions through their recognizer and their
metrics.

Two substitutions were needed on this machine, both documented in `scripts/reproduce_baseline.py`:

- The authors' code counts frames by globbing the extracted RGB jpgs of each test recording, which
  would mean extracting 24 GB of frames to obtain 32 integers. `tools/extract_labels.py` reads the
  frame count from the archive listing and pulls out only the label files. The count matches the
  frame count of the published mp4 for every recording checked.
- The metric uses a Cython package that needs a C compiler to install. `stepwitness/metrics.py` is a
  pure Python port of the weighted Damerau-Levenshtein distance it calls, with tests against an
  independent implementation on 2,000 random pairs and against the paper's own examples.
  `stepwitness/compat/weighted_levenshtein.py` serves it under the original import name, so the
  authors' metric code runs unmodified.

## Result

Reproduced on 2026-10-14 on all 32 test recordings (15 of them with errors), after the three test
archives verified against their published MD5s. Every row matches the paper's Table 4 to three
decimals, for all three baselines and both prediction sets.

| Baseline | Predictions | POS all | F1 all | Delay all (s) | POS errors | F1 errors | Delay errors (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| B1 (naive) | real-trained | 0.570 | 0.779 | 14.9 | 0.480 | 0.698 | 14.4 |
| B2 (confidence) | real-trained | 0.731 | 0.860 | 22.3 | 0.636 | 0.784 | 20.2 |
| B3 (expected) | real-trained | 0.797 | 0.883 | 22.4 | 0.731 | 0.816 | 20.4 |
| B1 (naive) | synthetic-only | 0.014 | 0.206 | 36.9 | 0.000 | 0.174 | 48.4 |
| B2 (confidence) | synthetic-only | 0.240 | 0.573 | 44.4 | 0.107 | 0.516 | 60.5 |
| B3 (expected) | synthetic-only | 0.597 | 0.734 | 49.5 | 0.571 | 0.731 | 71.4 |

Published B3 for comparison: real-trained 0.797 / 0.883 / 22.4 s on all, 0.731 / 0.816 / 20.4 s with
errors; synthetic-only 0.597 / 0.734 / 49.5 s on all, 0.571 / 0.731 / 71.4 s with errors. Identical.

Two things this establishes. The frame counts read from the archive listings are the counts the
authors' code would have globbed, or the delays would have drifted. And the pure Python metric port is
exact on real data, not only on the unit tests.

The per-recording breakdown is in `runs/phase0/baseline_results.json` (not committed); the table is
`runs/phase0/baseline_table.md`.
