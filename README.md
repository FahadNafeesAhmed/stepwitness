# Stepwitness

A procedure monitor built from a work instruction and the CAD files of the parts, with no labelled video.
It watches a worker, logs each completed step with a timestamp, and flags skipped or wrongly executed
steps. Every flag links to the frame and the instruction line it violates.

Evaluated on IndustReal (Schoonbeek et al., WACV 2024), the public industrial assembly benchmark with
labelled execution errors and published part geometry.

## Status

No result of this project has been measured yet. Code is in place for every phase; numbers follow.

| Phase | What | State |
|---|---|---|
| 0 | reproduce the published baseline from the authors' predictions | tooling ready, waiting on the test-label archives |
| 1 | procedure graph, LLM compiler, validation against the 22 labelled states | done, 10 tests |
| 2 | render factory: correct states and generated errors from the CAD | done, first batch rendering |
| 3 | verifier (frozen DINOv2, prototypes, head) and one-class localizer | code ready, training next |
| 4 | step recognizer and the PSR evaluation bridge | done, 7 tests |
| 5 | evidence timeline and viewer | done, 3 tests |
| 6 | edge profile (FP32, FP16, INT8 when TensorRT is available) | profiler ready |

## Layout

```
stepwitness/   the package: graph, renders, verifier, recognizer, timeline, edge
scripts/       entry points that run a phase end to end
tools/         one-off utilities (label extraction, checksums)
tests/         unit tests
docs/          design notes and measured results
data/          datasets and model files (not committed, see docs/data_manifest.md)
third_party/   reference code cloned read-only (not committed)
runs/          outputs of experiments (not committed)
```

## Setup

```
uv venv --python 3.11 --system-site-packages
uv pip install -e .
git clone https://github.com/TimSchoonbeek/IndustReal third_party/IndustReal
```

Torch, torchvision and OpenCV come from the system Python on the development machine.
