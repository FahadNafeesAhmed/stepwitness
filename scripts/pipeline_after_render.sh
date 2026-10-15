#!/usr/bin/env bash
# Phases 3 and 4 end to end, started once the render batch has finished: verifier, localizer,
# monitor over the test recordings, scores, flag scores, and three annotated clips for the page.
# Everything logs to runs/phase4/pipeline.log. Run from the repo root.
set -u
PY=.venv/Scripts/python.exe
LOG=runs/phase4/pipeline.log
mkdir -p runs/phase4
log () { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

until grep -qE "wrote manifest|Traceback|EXCEPTION" runs/phase2/render_v2_seed101.log || ! tasklist | grep -qi blender.exe; do sleep 20; done
log "render finished: $(ls data/renders/v2/images | wc -l) images"

log "verifier build"
$PY scripts/build_verifier.py --renders data/renders/v2 --out runs/phase3/verifier_v2 --epochs 40 >> "$LOG" 2>&1 || { log "verifier build FAILED"; exit 1; }
grep -E "^\[" "$LOG" | tail -3

log "localizer training"
$PY scripts/train_localizer.py --renders data/renders/v2 --out runs/phase3/localizer_v2 --model data/models/yolo/yolov8n.pt --epochs 15 --batch 16 >> "$LOG" 2>&1 || { log "localizer training FAILED"; exit 1; }

log "monitor over the test recordings"
$PY scripts/run_monitor.py --verifier runs/phase3/verifier_v2 --localizer runs/phase3/localizer_v2/train/weights/best.pt --out runs/phase4/monitor_v2 --stride 1 >> "$LOG" 2>&1 || { log "monitor FAILED"; exit 1; }

log "scores"
$PY scripts/evaluate_psr.py --predictions runs/phase4/monitor_v2 --out runs/phase4/monitor_v2 >> "$LOG" 2>&1
$PY scripts/evaluate_flags.py --events runs/phase4/monitor_v2 --out runs/phase4/monitor_v2 >> "$LOG" 2>&1
cat runs/phase4/monitor_v2/psr_table_row.md | tee -a "$LOG"

log "clips"
for rec in 03_assy_1_3 08_assy_2_4 13_main_1_3; do
  $PY scripts/make_videos.py monitor --events runs/phase4/monitor_v2 --video "data/industreal/videos/$rec.mp4" --out "site/assets/videos/$rec.mp4" --speed 2 >> "$LOG" 2>&1 || log "clip $rec failed"
done
log "pipeline done"
