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

## The site

`site/` is a static page in the style of a long-form write-up, with the architecture, the figures,
a frame stepper over real test recordings, and the clips. Deploy on Vercel by importing the repo:
`vercel.json` points the deployment at `site/` with no build step. Locally:

```
python -m http.server 8787 --directory site
```

Clips and stepper data are produced by `scripts/make_videos.py` and `tools/export_stepper.py` from
the monitor's event files.

## Running each phase

```
# 0  reproduce the published baseline (needs the test label files, see docs/data_manifest.md)
python tools/extract_labels.py data/industreal/zips/test_p1.zip data/industreal/zips/test_p2.zip data/industreal/zips/test_p3.zip
python scripts/reproduce_baseline.py --out runs/phase0

# 1  compile an instruction sheet into a graph with a local LLM and check it against the labelled states
python scripts/compile_procedure.py --sheet docs/instructions/industreal_assembly.md --name industreal_assembly --start-state 10000000000 --labelled 1-13

# 2  render the training set (Blender, GPU)
third_party/blender/blender-4.5.14-windows-x64/blender.exe -b --python tools/blender_render_dataset.py --     --fbx-dir data/industreal/part_geometries/part_geometries --out data/renders/v2 --states all --per-state 600 --error-fraction 0.5 --seed 100

# 3  build the verifier and train the localizer
python scripts/build_verifier.py --renders data/renders/v2 --out runs/phase3/verifier_v2
python scripts/train_localizer.py --renders data/renders/v2 --out runs/phase3/localizer_v2 --model data/models/yolo/yolov8n.pt --epochs 20

# 4  run the monitor on the test recordings and score it with the authors' metric code
python scripts/run_monitor.py --verifier runs/phase3/verifier_v2 --localizer runs/phase3/localizer_v2/train/weights/best.pt --out runs/phase4/monitor_v2
python scripts/evaluate_psr.py --predictions runs/phase4/monitor_v2 --out runs/phase4/monitor_v2

# 5  review the evidence timeline in a browser
python scripts/serve_timeline.py --events runs/phase4/monitor_v2

# 6  measure the deployed cost
python scripts/profile_edge.py --video data/industreal/videos/03_assy_0_1.mp4 --verifier runs/phase3/verifier_v2 --localizer runs/phase3/localizer_v2/train/weights/best.pt --out runs/phase6/profile_v2
```
