"""Phase 5: the evidence timeline viewer.

A small FastAPI service over the event files written by scripts/run_monitor.py. The page lists
recordings, shows each one's completions and flags on a timeline, and opens the frame and the
instruction line behind any event. Frames are pulled from the mp4 on demand. Nothing is acted on.

Usage:
    python scripts/serve_timeline.py --events runs/phase4/monitor_v2 --videos data/industreal/videos --port 8765
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stepwitness.procedure import Procedure  # noqa: E402
from stepwitness.timeline import summary  # noqa: E402

PAGE = """<!doctype html><meta charset="utf-8"><title>Stepwitness timeline</title>
<style>
body{font-family:"Times New Roman",Times,serif;margin:24px;color:#111;background:#fff}
h1{font-weight:400;font-size:28px;margin:0 0 6px}
.sub{color:#555;font-size:14px;margin-bottom:18px}
select{font:inherit;padding:4px 8px}
table{border-collapse:collapse;width:100%;margin-top:14px;font-size:15px;border-top:2px solid #111;border-bottom:2px solid #111}
th{text-align:left;border-bottom:1px solid #111;padding:6px 8px 6px 0}
td{padding:6px 8px 6px 0;border-bottom:1px solid #ccc;vertical-align:top}
tr.flag td{color:#b3261e}
.bar{position:relative;height:28px;border-top:1px solid #111;border-bottom:1px solid #111;margin:14px 0}
.tick{position:absolute;top:0;width:2px;height:28px;background:#111}
.tick.flag{background:#b3261e}
#frame{max-width:100%;border:1px solid #111;margin-top:10px}
.mono{font-family:Consolas,monospace;font-size:13px}
.sum{display:flex;gap:28px;flex-wrap:wrap;font-size:14px;color:#333;margin-top:8px}
</style>
<h1>Stepwitness timeline</h1>
<div class="sub">Evidence for review. Nothing here acts on the line.</div>
<label>Recording <select id="rec"></select></label>
<div class="sum" id="sum"></div>
<div class="bar" id="bar"></div>
<table><thead><tr><th>Time</th><th>Frame</th><th>Event</th><th>Step</th><th>Instruction</th><th>Confidence</th></tr></thead><tbody id="rows"></tbody></table>
<img id="frame" alt="">
<div class="mono" id="evidence"></div>
<script>
const recSel=document.getElementById('rec');
async function loadList(){const r=await fetch('/api/recordings');const names=await r.json();
 for(const n of names){const o=document.createElement('option');o.value=n;o.textContent=n;recSel.appendChild(o)}
 if(names.length)load(names[0]);}
recSel.onchange=()=>load(recSel.value);
function fmt(f){const s=f/10;const m=Math.floor(s/60);return m+':'+String(Math.floor(s%60)).padStart(2,'0')}
async function load(name){const r=await fetch('/api/events/'+name);const d=await r.json();
 const rows=document.getElementById('rows');rows.innerHTML='';
 const bar=document.getElementById('bar');bar.innerHTML='';
 const last=d.last_frame||1;
 document.getElementById('sum').innerHTML=`<span>completions ${d.summary.completions}</span><span>flags ${d.summary.flags}</span><span>cycle time ${d.summary.cycle_time_s==null?'n/a':d.summary.cycle_time_s.toFixed(1)+' s'}</span><span>recording ${d.summary.recording_s.toFixed(1)} s</span>`;
 for(const e of d.events){if(e.type==='hold')continue;
  const tr=document.createElement('tr');const isFlag=['omission','execution_error','wrong_order'].includes(e.type);if(isFlag)tr.className='flag';
  tr.innerHTML=`<td>${fmt(e.frame)}</td><td>${e.frame}</td><td>${e.type}${e.at_end?' (at end)':''}</td><td>${e.step??''}</td><td>${e.line?('line '+e.line+': '):''}${e.text||d.lines[e.line]||''}</td><td>${e.confidence!=null?Number(e.confidence).toFixed(2):''}</td>`;
  tr.onclick=()=>show(name,e);rows.appendChild(tr);
  const t=document.createElement('div');t.className='tick'+(isFlag?' flag':'');t.style.left=(100*e.frame/last)+'%';t.title=e.type+' @ '+fmt(e.frame);bar.appendChild(t);}
}
function show(name,e){document.getElementById('frame').src='/api/frame/'+name+'/'+e.frame+'?box='+encodeURIComponent(JSON.stringify((e.evidence||{}).box||null));
 document.getElementById('evidence').textContent=JSON.stringify(e.evidence||{},null,1);}
loadList();
</script>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--videos", type=Path, default=ROOT / "data" / "industreal" / "videos")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response
    import uvicorn

    app = FastAPI(title="Stepwitness timeline")

    def lines_for(name: str) -> dict[int, str]:
        proc = Procedure.builtin("industreal_assembly" if "assy" in name else "industreal_maintenance")
        return {s.line: s.text for s in proc.steps.values()}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/recordings")
    def recordings():
        return sorted(p.name.replace(".events.json", "") for p in args.events.glob("*.events.json"))

    @app.get("/api/events/{name}")
    def events(name: str):
        f = args.events / f"{name}.events.json"
        if not f.exists():
            raise HTTPException(404)
        ev = json.loads(f.read_text())
        last = max((e["frame"] for e in ev), default=0)
        cap = cv2.VideoCapture(str(args.videos / f"{name}.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or last
        cap.release()
        return {"events": ev, "last_frame": n, "summary": summary(ev, 0, n), "lines": lines_for(name)}

    @app.get("/api/frame/{name}/{frame}")
    def frame(name: str, frame: int, box: str = "null"):
        cap = cv2.VideoCapture(str(args.videos / f"{name}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(404)
        try:
            b = json.loads(box)
        except Exception:
            b = None
        if b:
            x, y, w, h = [int(v) for v in b]
            cv2.rectangle(img, (x, y), (x + w, y + h), (30, 200, 30), 3)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
