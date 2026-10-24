"""
build_site.py - turn predictions.db into a static page (docs/index.html) plus a downloadable copy of the database.

No server and no framework: GitHub Pages serves docs/. The page shows today's open questions, locked and resolved
markets, and the running Brier/accuracy. Visitors can also enter their own probability for an open question; it is
saved in THEIR browser only (localStorage) and scored against the model once the market resolves. No accounts.

The output depends only on the database (no build timestamp), so an unchanged database gives an unchanged page
and the scheduled workflow makes no commit.

Usage:  python build_site.py --db data/predictions.db --out docs
"""

import argparse
import json
import os
import shutil
import sqlite3

COLUMNS = ["id", "matchId", "team1", "team2", "seriesName", "match_format", "match_start_time", "lock_time",
           "market_status", "venue", "question", "probability", "probability_reasoning", "resolution_rule",
           "reasoning", "actual_outcome", "is_correct", "created_at", "brier_score"]


def read_rows(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(f"SELECT {', '.join(COLUMNS)} FROM predictions ORDER BY match_start_time DESC, id DESC")]
    conn.close()
    return rows


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cricket Prediction Engine</title>
<style>
  :root { --bg:#fbfaf8; --card:#ffffff; --ink:#1d1d1b; --muted:#6b6a66; --line:#e6e3dc; --accent:#1f6f4a; --bad:#b4432f; --bar:#d9e8df; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141412; --card:#1c1c1a; --ink:#ecebe6; --muted:#9a998f; --line:#2c2c29; --accent:#5fbf8f; --bad:#e07a66; --bar:#23382d; } }
  :root[data-theme="dark"] { --bg:#141412; --card:#1c1c1a; --ink:#ecebe6; --muted:#9a998f; --line:#2c2c29; --accent:#5fbf8f; --bad:#e07a66; --bar:#23382d; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 28px 16px 64px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  h2 { font-size: 1.05rem; margin: 32px 0 10px; letter-spacing: .02em; }
  p.sub { color:var(--muted); margin:0 0 18px; }
  .stats { display:grid; grid-template-columns: repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin: 16px 0 6px; }
  .stat { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
  .stat b { display:block; font-size:1.35rem; }
  .stat span { color:var(--muted); font-size:.82rem; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin: 10px 0; }
  .meta { color:var(--muted); font-size:.85rem; }
  .q { font-weight:600; margin: 4px 0 8px; }
  .bar { height:10px; background:var(--bar); border-radius:6px; overflow:hidden; margin: 6px 0 2px; }
  .bar > i { display:block; height:100%; background:var(--accent); }
  .row { display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; font-size:.9rem; }
  details { margin-top:8px; } summary { cursor:pointer; color:var(--muted); font-size:.88rem; }
  details p { margin:6px 0; font-size:.9rem; }
  .guess { margin-top:10px; padding-top:10px; border-top:1px dashed var(--line); font-size:.9rem; }
  .guess input[type=range] { width: min(320px,100%); vertical-align: middle; }
  button { font:inherit; border:1px solid var(--line); background:var(--bg); color:var(--ink); border-radius:8px; padding:4px 12px; cursor:pointer; }
  button:hover { border-color: var(--accent); }
  table { width:100%; border-collapse:collapse; font-size:.88rem; }
  th, td { text-align:left; padding:7px 6px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--muted); font-weight:500; }
  .ok { color:var(--accent); } .no { color:var(--bad); }
  .scroll { overflow-x:auto; }
  .empty { color:var(--muted); padding:10px 0; }
  code { background:var(--card); border:1px solid var(--line); border-radius:5px; padding:1px 5px; font-size:.85em; }
  a { color: var(--accent); }
</style>
</head>
<body>
<main>
  <h1>Cricket Prediction Engine</h1>
  <p class="sub">An LLM (DeepSeek) writes a yes/no question and a win probability for each upcoming match before it starts. Markets lock <span id="lockmin"></span> minutes before the start, then resolve from the official result. Every number below comes from the database in this repo.</p>

  <div class="stats" id="stats"></div>

  <h2>Open markets</h2>
  <div id="open"></div>

  <h2>Locked, waiting for the result</h2>
  <div id="locked"></div>

  <h2>Resolved</h2>
  <div class="card scroll" id="resolved"></div>

  <h2>The database</h2>
  <div class="card">
    <p style="margin-top:0">Everything is in one SQLite file, committed to the repo by the scheduled run: <a href="predictions.db" download>download predictions.db</a>.</p>
    <p class="meta">Table <code>predictions</code>: matchId, team1, team2, seriesName, match_format, match_start_time, lock_time, market_status (OPEN / LOCKED / RESOLVED / VOID), question, probability, probability_reasoning, resolution_rule, reasoning, actual_outcome, is_correct, brier_score, created_at.</p>
    <p class="meta" style="margin-bottom:0">Try: <code>sqlite3 predictions.db "select question, probability, actual_outcome, brier_score from predictions where market_status='RESOLVED'"</code></p>
  </div>
  <p class="meta" style="margin-top:28px">Probabilities are the model's raw output (no calibration applied here). Brier score = (probability - outcome)<sup>2</sup>; always saying 50% scores 0.25. Your guesses are stored in your own browser only.</p>
</main>

<script id="data" type="application/json">__DATA__</script>
<script>
const LOCK_MIN = __LOCK_MIN__;
const rows = JSON.parse(document.getElementById('data').textContent);
const $ = id => document.getElementById(id);
$('lockmin').textContent = LOCK_MIN;

// --- guesses live in this browser only; every access is guarded because storage can be blocked
const KEY = 'cricket-guesses-v1';
function loadGuesses() { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { return {}; } }
function saveGuess(id, p) { try { const g = loadGuesses(); g[id] = p; localStorage.setItem(KEY, JSON.stringify(g)); return true; } catch (e) { return false; } }

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = p => (p * 100).toFixed(0) + '%';
const when = iso => iso ? new Date(iso).toLocaleString([], {dateStyle:'medium', timeStyle:'short'}) : '';
function until(iso) {
  const ms = new Date(iso) - Date.now();
  if (ms <= 0) return 'locked';
  const m = Math.floor(ms / 60000), d = Math.floor(m / 1440), h = Math.floor((m % 1440) / 60);
  return 'locks in ' + (d ? d + 'd ' : '') + (h ? h + 'h ' : '') + (m % 60) + 'm';
}
const isOpen = r => r.market_status === 'OPEN' && new Date(r.lock_time) > Date.now();

function stats() {
  const done = rows.filter(r => r.market_status === 'RESOLVED');
  const acc = done.length ? done.filter(r => r.is_correct).length / done.length : null;
  const brier = done.length ? done.reduce((s, r) => s + r.brier_score, 0) / done.length : null;
  const cells = [
    [rows.length, 'markets created'], [rows.filter(isOpen).length, 'open now'],
    [done.length, 'resolved'], [acc === null ? '–' : pct(acc), 'accuracy so far'],
    [brier === null ? '–' : brier.toFixed(3), 'mean Brier (50% = 0.25)']
  ];
  $('stats').innerHTML = cells.map(c => `<div class="stat"><b>${c[0]}</b><span>${c[1]}</span></div>`).join('');
}

function card(r, extra) {
  return `<div class="card">
    <div class="meta">${esc(r.team1)} vs ${esc(r.team2)} · ${esc(r.match_format)} · ${esc(r.seriesName)}<br>${when(r.match_start_time)} · ${esc(r.venue)}</div>
    <div class="q">${esc(r.question)}</div>
    <div class="row"><span>Model: <b>${pct(r.probability)}</b> YES</span><span class="meta">${extra || ''}</span></div>
    <div class="bar"><i style="width:${r.probability * 100}%"></i></div>
    <details><summary>Why</summary><p>${esc(r.probability_reasoning)}</p><p class="meta">Resolution: ${esc(r.resolution_rule)}</p></details>
    ${r._tail || ''}
  </div>`;
}

function openMarkets() {
  const g = loadGuesses();
  const list = rows.filter(isOpen).sort((a, b) => new Date(a.match_start_time) - new Date(b.match_start_time));
  if (!list.length) { $('open').innerHTML = '<div class="empty">No open markets right now. New fixtures are picked up a few times a day.</div>'; return; }
  $('open').innerHTML = list.map(r => {
    const mine = g[r.matchId];
    r._tail = `<div class="guess">Your probability of YES: <input type="range" min="1" max="99" value="${mine ? Math.round(mine * 100) : 50}" data-id="${esc(r.matchId)}"> <b class="gv">${mine ? pct(mine) : '50%'}</b> <button data-save="${esc(r.matchId)}">${mine ? 'Update' : 'Save'}</button> <span class="meta saved"></span></div>`;
    return card(r, until(r.lock_time));
  }).join('');
  document.querySelectorAll('input[type=range]').forEach(el => el.addEventListener('input', () => { el.parentNode.querySelector('.gv').textContent = el.value + '%'; }));
  document.querySelectorAll('button[data-save]').forEach(b => b.addEventListener('click', () => {
    const range = b.parentNode.querySelector('input'); const ok = saveGuess(b.dataset.save, +range.value / 100);
    b.parentNode.querySelector('.saved').textContent = ok ? 'saved in this browser' : 'could not save (storage blocked)';
  }));
}

function lockedMarkets() {
  const list = rows.filter(r => r.market_status === 'LOCKED' || (r.market_status === 'OPEN' && !isOpen(r)));
  $('locked').innerHTML = list.length ? list.map(r => card(r, 'result pending')).join('') : '<div class="empty">None.</div>';
}

function resolvedTable() {
  const g = loadGuesses();
  const list = rows.filter(r => r.market_status === 'RESOLVED' || r.market_status === 'VOID');
  if (!list.length) { $('resolved').innerHTML = '<div class="empty">Nothing has resolved yet.</div>'; return; }
  let mine = 0, model = 0, n = 0;
  const body = list.map(r => {
    const y = r.actual_outcome === 'YES' ? 1 : 0, guess = g[r.matchId];
    let you = '';
    if (guess !== undefined && r.market_status === 'RESOLVED') { const b = (guess - y) ** 2; mine += b; model += r.brier_score; n++; you = pct(guess) + ' (' + b.toFixed(3) + ')'; }
    const res = r.market_status === 'VOID' ? 'void' : `<span class="${r.is_correct ? 'ok' : 'no'}">${esc(r.actual_outcome)} ${r.is_correct ? '✓' : '✗'}</span>`;
    return `<tr><td>${esc(r.question)}<div class="meta">${when(r.match_start_time)}</div></td><td>${pct(r.probability)}</td><td>${res}</td><td>${r.brier_score == null ? '' : r.brier_score.toFixed(3)}</td><td>${you}</td></tr>`;
  }).join('');
  const foot = n ? `<p class="meta">On the ${n} resolved market(s) you guessed: your mean Brier ${(mine / n).toFixed(3)} vs the model's ${(model / n).toFixed(3)} (lower is better).</p>` : '';
  $('resolved').innerHTML = `<table><thead><tr><th>Question</th><th>Model</th><th>Result</th><th>Brier</th><th>You (Brier)</th></tr></thead><tbody>${body}</tbody></table>${foot}`;
}

stats(); openMarkets(); lockedMarkets(); resolvedTable();
setInterval(() => { openMarkets(); stats(); }, 60000);
</script>
</body>
</html>
"""


def build(db_path: str, out_dir: str, lock_minutes: int) -> int:
    rows = read_rows(db_path)
    os.makedirs(out_dir, exist_ok=True)
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")   # keep "</script>" in text from closing the tag
    html = TEMPLATE.replace("__DATA__", data).replace("__LOCK_MIN__", str(lock_minutes))
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    shutil.copyfile(db_path, os.path.join(out_dir, "predictions.db"))
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Build the static page from predictions.db.")
    ap.add_argument("--db", default="data/predictions.db")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--lock-minutes", type=int, default=35, help="shown in the page text; keep equal to LOCK_MINUTES_BEFORE_START")
    args = ap.parse_args()
    n = build(args.db, args.out, args.lock_minutes)
    print(f"Wrote {args.out}/index.html ({n} markets) and {args.out}/predictions.db")


if __name__ == "__main__":
    main()