#!/usr/bin/env bash
# After pipeline_v3.sh: replay the cached verifier output with the procedure guards on, score both
# settings, make the clips and the stepper data from the guarded replay, write the numbers into the
# page, and profile the edge cost (GPU, one job). Run from the repo root.
set -u
PY=.venv/Scripts/python.exe
LOG=runs/phase4/finish_v3.log
log () { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
GUARD="--reach 4 --stable-frames 5 --release-frames 50 --error-frames 20 --error-margin 0.4"

log "replay with guards on: $GUARD"
$PY scripts/rescore.py --posteriors runs/phase4/monitor_v3 --out runs/phase4/monitor_v3_guarded --label "Stepwitness, guards on" $GUARD >> "$LOG" 2>&1 || { log "guarded replay FAILED"; exit 1; }
log "replay with guards off (B3 constants), as a check that the replay equals the live run"
$PY scripts/rescore.py --posteriors runs/phase4/monitor_v3 --out runs/phase4/monitor_v3_plain --label "Stepwitness, guards off" >> "$LOG" 2>&1 || { log "plain replay FAILED"; exit 1; }
$PY - <<'EOF' | tee -a "$LOG"
import json, glob, os
same = all(json.load(open(p)) == json.load(open(p.replace("monitor_v3_plain", "monitor_v3"))) for p in glob.glob("runs/phase4/monitor_v3_plain/*.yhat.json"))
print("plain replay identical to the live run:", same)
EOF
cat runs/phase4/monitor_v3_guarded/psr_table_row.md runs/phase4/monitor_v3_plain/psr_table_row.md | tee -a "$LOG"

log "sensitivity of the guard settings (release 50 vs 100, reach 3 vs 4)"
for cfg in "r3_rel50:--reach 3 --stable-frames 5 --release-frames 50" "r4_rel100:--reach 4 --stable-frames 5 --release-frames 100" "r4_rel200:--reach 4 --stable-frames 5 --release-frames 200"; do
  name="${cfg%%:*}"; args="${cfg#*:}"
  $PY scripts/rescore.py --posteriors runs/phase4/monitor_v3 --out "runs/phase4/sensitivity/$name" --label "$name" $args --error-frames 20 --error-margin 0.4 > /dev/null 2>&1
  cat "runs/phase4/sensitivity/$name/psr_table_row.md" | tee -a "$LOG"
done

log "clips and stepper data from the guarded replay"
rm -rf site/assets/stepper
for rec in 03_assy_1_3 08_assy_2_4 13_main_1_3; do
  $PY scripts/make_videos.py monitor --events runs/phase4/monitor_v3_guarded --video "data/industreal/videos/$rec.mp4" --out "site/assets/videos/$rec.mp4" --speed 2 >> "$LOG" 2>&1 || log "clip $rec failed"
  $PY tools/export_stepper.py --events runs/phase4/monitor_v3_guarded --video "data/industreal/videos/$rec.mp4" --label "$rec" --stride 25 --width 800 >> "$LOG" 2>&1 || log "stepper $rec failed"
done

log "numbers into the page"
$PY tools/fill_site_results.py --run runs/phase4/monitor_v3_guarded --run-plain runs/phase4/monitor_v3_plain >> "$LOG" 2>&1 || log "fill FAILED"

log "edge profile (FP32 with autocast off, FP16)"
$PY scripts/profile_edge.py --video data/industreal/videos/03_assy_0_1.mp4 --verifier runs/phase3/verifier_v3 --verifier-kind v3 --localizer runs/phase3/localizer_v2/train/weights/best.pt --out runs/phase6/profile_v3 --frames 300 >> "$LOG" 2>&1 || log "profile FAILED"
log "finish v3 done"
