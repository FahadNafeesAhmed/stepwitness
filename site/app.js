// Stepwitness page: the frame stepper (one recording, frame by frame, timeline filling in).
// Data: assets/stepper/index.json lists recordings; each has a folder with frames and events.
(function () {
  const root = document.getElementById('stepper');
  if (!root) return;
  const img = root.querySelector('img');
  const box = root.querySelector('.box');
  const hud = root.querySelector('.hud');
  const bar = root.querySelector('.bar');
  const cursor = root.querySelector('.cursor');
  const status = root.querySelector('.status');
  const eventsEl = root.querySelector('.events');
  const sel = root.querySelector('select');
  const btnPrev = root.querySelector('[data-step="-1"]');
  const btnNext = root.querySelector('[data-step="1"]');
  const btnPlay = root.querySelector('[data-play]');

  let rec = null, i = 0, timer = null;

  function fmt(f) { const s = f / 10; const m = Math.floor(s / 60); return m + ':' + String(Math.floor(s % 60)).padStart(2, '0'); }

  function render() {
    if (!rec) return;
    const fr = rec.frames[i];
    img.src = rec.base + fr.file;
    const [w, h] = rec.size;
    if (fr.box) {
      const [x, y, bw, bh] = fr.box;
      box.style.display = 'block';
      box.style.left = (100 * x / w) + '%';
      box.style.top = (100 * y / h) + '%';
      box.style.width = (100 * bw / w) + '%';
      box.style.height = (100 * bh / h) + '%';
    } else box.style.display = 'none';
    hud.textContent = `frame ${fr.frame}  ·  ${fmt(fr.frame)}` + (fr.top ? `  ·  ${fr.top}  p=${fr.top_p.toFixed(2)}` : '');
    cursor.style.left = (100 * fr.frame / rec.n_frames) + '%';
    for (const t of bar.querySelectorAll('.tick')) t.classList.toggle('done', Number(t.dataset.frame) <= fr.frame);
    const shown = rec.events.filter(e => e.frame <= fr.frame && e.type !== 'hold');
    eventsEl.innerHTML = shown.slice(-6).reverse().map(e => `<div class="${e.type}">${fmt(e.frame)}  ${e.text}</div>`).join('') || '<div class="muted">no events yet</div>';
    status.textContent = `${i + 1} / ${rec.frames.length}`;
  }

  function load(r) {
    rec = r; i = 0;
    bar.querySelectorAll('.tick').forEach(t => t.remove());
    for (const e of r.events) {
      if (e.type === 'hold') continue;
      const t = document.createElement('div');
      t.className = 'tick ' + (e.type === 'completion' ? 'completion' : 'flagged');
      t.dataset.frame = e.frame;
      t.style.left = (100 * e.frame / r.n_frames) + '%';
      t.title = `${e.type} @ ${fmt(e.frame)}`;
      bar.appendChild(t);
    }
    render();
  }

  function step(d) { if (!rec) return; i = Math.max(0, Math.min(rec.frames.length - 1, i + d)); render(); }
  function toggle() {
    if (timer) { clearInterval(timer); timer = null; btnPlay.textContent = '▶'; return; }
    btnPlay.textContent = '❚❚';
    timer = setInterval(() => { if (i >= rec.frames.length - 1) { toggle(); return; } step(1); }, 400);
  }
  btnPrev.addEventListener('click', () => step(-1));
  btnNext.addEventListener('click', () => step(1));
  btnPlay.addEventListener('click', toggle);
  document.addEventListener('keydown', e => { if (e.key === 'ArrowRight') step(1); if (e.key === 'ArrowLeft') step(-1); });

  fetch('assets/stepper/index.json').then(r => r.ok ? r.json() : null).then(list => {
    if (!list || !list.length) { const fig = document.getElementById('stepper-fig'); if (fig) fig.hidden = true; const note = document.getElementById('stepper-note'); if (note) note.hidden = false; return; }
    for (const item of list) { const o = document.createElement('option'); o.value = item.path; o.textContent = item.label; sel.appendChild(o); }
    const pick = () => fetch(sel.value + '/index.json').then(r => r.json()).then(r => { r.base = sel.value + '/'; load(r); });
    sel.addEventListener('change', pick);
    pick();
  }).catch(() => { const fig = document.getElementById('stepper-fig'); if (fig) fig.hidden = true; const note = document.getElementById('stepper-note'); if (note) note.hidden = false; });
})();
