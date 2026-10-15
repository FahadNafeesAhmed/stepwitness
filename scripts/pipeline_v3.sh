#!/usr/bin/env bash
# Second pass of phases 3 and 4, one GPU process at a time, written for a machine with little free RAM:
#   localizer (no data-loader workers) -> verifier v3 (patch tokens, two heads) -> monitor with v3 over
#   all 32 test recordings -> scores -> flag scores -> clips and stepper data for the page.
# Logs to runs/phase4/pipeline_v3.log. Run from the repo root.
set -u
PY=.venv/Scripts/python.exe
LOG=runs/phase4/pipeline_v3.log
mkdir -p runs/phase4
log () { echo "[$(date +%T)] $*" | tee -a "$LOG"; }

if [ ! -f runs/phase3/localizer_v2/train/weights/best.pt ]; then
  log "localizer training (workers=0)"
  $PY scripts/train_localizer.py --renders data/renders/v2 --out runs/phase3/localizer_v2 --model data/models/yolo/yolov8n.pt --epochs 12 --batch 16 --workers 0 >> "$LOG" 2>&1 || { log "localizer training FAILED"; exit 1; }
else
  log "localizer already trained"
fi

if [ ! -f runs/phase3/verifier_v3/head_v3.pt ]; then
  log "verifier v3 build"
  $PY scripts/build_verifier_v3.py --renders data/renders/v2 --out runs/phase3/verifier_v3 --image-size 448 --grid 8 --epochs 30 >> "$LOG" 2>&1 || { log "verifier v3 build FAILED"; exit 1; }
  grep -E "final|state_acc|err_recall" "$LOG" | tail -3
else
  log "verifier v3 already built"
fi

log "monitor (v3) over the test recordings"
$PY scripts/run_monitor.py --verifier runs/phase3/verifier_v3 --verifier-kind v3 --localizer runs/phase3/localizer_v2/train/weights/best.pt --out runs/phase4/monitor_v3 --stride 1 >> "$LOG" 2>&1 || { log "monitor FAILED"; exit 1; }

log "scores"
$PY scripts/evaluate_psr.py --predictions runs/phase4/monitor_v3 --out runs/phase4/monitor_v3 --label "Stepwitness v3" >> "$LOG" 2>&1
$PY scripts/evaluate_flags.py --events runs/phase4/monitor_v3 --out runs/phase4/monitor_v3 >> "$LOG" 2>&1
cat runs/phase4/monitor_v3/psr_table_row.md | tee -a "$LOG"

log "clips and stepper data"
for rec in 03_assy_1_3 08_assy_2_4 13_main_1_3; do
  $PY scripts/make_videos.py monitor --events runs/phase4/monitor_v3 --video "data/industreal/videos/$rec.mp4" --out "site/assets/videos/$rec.mp4" --speed 2 >> "$LOG" 2>&1 || log "clip $rec failed"
  $PY tools/export_stepper.py --events runs/phase4/monitor_v3 --video "data/industreal/videos/$rec.mp4" --label "$rec" --stride 25 --width 800 >> "$LOG" 2>&1 || log "stepper $rec failed"
done
log "pipeline v3 done"
