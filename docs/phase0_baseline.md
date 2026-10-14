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

Filled in from `runs/phase0/baseline_table.md` once all three test archives are verified. The
partial run on the first nine recordings (`runs/phase0_partial`) executes end to end; its numbers are
not reported because the published figures are averages over all 32 test recordings.

| Baseline | Predictions | POS all | F1 all | Delay all (s) | POS errors | F1 errors | Delay errors (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| B3, published | real-trained | 0.797 | 0.883 | 22.4 | 0.731 | 0.816 | 20.4 |
| B3, reproduced | real-trained | pending | | | | | |
| B3, published | synthetic-only | 0.597 | 0.734 | 49.5 | 0.571 | 0.731 | 71.4 |
| B3, reproduced | synthetic-only | pending | | | | | |
