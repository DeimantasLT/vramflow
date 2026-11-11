// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// OGraf grafikos renderis: jau įdiegtas Microsoft Edge (headless) per CDP, permatomas fonas, kadrai -> stdout kaip
// neapdoroti RGBA baitai (W*H*4, straight alfa). Be išorinių paketų (tik Node >= 22). Naudojimas:
//   node render_edge.mjs [--seconds 8] [--w 1920] [--h 1080] [--edge <msedge.exe>] [--stop-on-stdin-eof] > frames.rgba
import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { inflateSync } from 'node:zlib';

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const W = +arg('w', 1920), H = +arg('h', 1080), SECONDS = +arg('seconds', 8);
const EDGE = arg('edge', 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe');
const root = join(dirname(fileURLToPath(import.meta.url)), 'graphic');
const types = { '.html': 'text/html', '.mjs': 'text/javascript', '.json': 'application/json' };
const srv = createServer((q, r) => {
  try {
    const f = q.url.split('?')[0].replace(/^\/+/, '') || 'host.html';
    if (f.includes('..')) throw 0;
    const body = readFileSync(join(root, f));
    r.writeHead(200, { 'content-type': types[extname(f)] || 'text/plain' });
    r.end(body);
  } catch { r.writeHead(404); r.end(); }
}).listen(0, '127.0.0.1');
await new Promise(r => srv.once('listening', r));
const port = srv.address().port;

function png(buf) { // 8 bitų RGBA be interlace (tokį duoda Chromium)
  let p = 8, w = 0, h = 0; const idat = [];
  while (p < buf.length) {
    const n = buf.readUInt32BE(p), t = buf.toString('latin1', p + 4, p + 8);
    if (t === 'IHDR') {
      w = buf.readUInt32BE(p + 8); h = buf.readUInt32BE(p + 12);
      if (buf[p + 16] !== 8 || buf[p + 17] !== 6 || buf[p + 20] !== 0) throw new Error('netikėtas PNG formatas');
    }
    if (t === 'IDAT') idat.push(buf.subarray(p + 8, p + 8 + n));
    p += 12 + n;
  }
  const raw = inflateSync(Buffer.concat(idat)), st = w * 4, out = Buffer.alloc(h * st);
  for (let y = 0; y < h; y++) {
    const ft = raw[y * (st + 1)], s = y * (st + 1) + 1, o = y * st;
    for (let x = 0; x < st; x++) {
      const a = x >= 4 ? out[o + x - 4] : 0, b = y ? out[o - st + x] : 0, c = x >= 4 && y ? out[o - st + x - 4] : 0;
      let v = raw[s + x];
      if (ft === 1) v += a; else if (ft === 2) v += b; else if (ft === 3) v += (a + b) >> 1;
      else if (ft === 4) { const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c); v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c; }
      out[o + x] = v & 255;
    }
  }
  return { w, h, rgba: out };
}

const ud = mkdtempSync(join(tmpdir(), 'vf-edge-'));
let edge = null, cleaned = false;
// Valymas: Edge procesų medis (taskkill /T) ir laikinas profilio katalogas; kviečiamas visais išėjimo keliais.
function cleanup() {
  if (cleaned) return; cleaned = true;
  try { srv.close(); } catch {}
  // msedge.exe paleidėjas grįžta iškart, o tikri procesai gyvena toliau -> naikinami visi procesai, kurių komandinėje
  // eilutėje yra šio paleidimo profilio katalogas (PID medis čia nepatikimas).
  try {
    spawnSync('powershell.exe', ['-NoProfile', '-Command',
      `Get-CimInstance Win32_Process -Filter "Name='msedge.exe'" | Where-Object { $_.CommandLine -like '*${ud.replace(/'/g, "''")}*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`],
      { stdio: 'ignore', timeout: 15000 });
  } catch {}
  for (let i = 0; i < 40; i++) { // Edge procesai gali dar trumpai laikyti failus (iki ~8 s)
    try { rmSync(ud, { recursive: true, force: true }); break; } catch { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 200); }
  }
}
function die(code, msg) { if (msg) console.error('render_edge: ' + msg); cleanup(); process.exit(code); }
for (const s of ['SIGINT', 'SIGTERM', 'SIGBREAK']) process.on(s, () => die(0, 'signalas ' + s));
process.on('exit', cleanup);
process.on('uncaughtException', e => die(1, String(e && e.stack || e)));
process.on('unhandledRejection', e => die(1, String(e && e.stack || e)));
// --stop-on-stdin-eof: stdin EOF = sustabdyti (paleidimo skriptas taip nutraukia renderį; WSL->Windows vamzdyje EPIPE nepatikimas)
if (process.argv.includes('--stop-on-stdin-eof')) { process.stdin.on('end', () => die(0, 'stdin uždarytas')); process.stdin.resume(); }
process.stdout.on('error', e => die(e.code === 'EPIPE' ? 0 : 1, e.code === 'EPIPE' ? 'stdout uždarytas (vartotojas baigė)' : String(e)));

edge = spawn(EDGE, ['--headless=new', '--disable-gpu', '--hide-scrollbars', '--no-first-run', '--remote-debugging-port=0',
  `--user-data-dir=${ud}`, `--window-size=${W},${H}`, 'about:blank'], { stdio: 'ignore' });
edge.on('error', e => die(1, 'Edge nepasileido: ' + e.message));
// (paleidėjo procesas baigiasi iškart — jo 'exit' nelaikomas klaida; nutrūkus CDP ryšiui baigiama per ws 'close')
// Windows GUI procesas stderr į vamzdį nerašo -> DevTools adresas imamas iš <user-data-dir>/DevToolsActivePort.
const wsUrl = await new Promise((res, rej) => {
  const t0 = Date.now();
  const iv = setInterval(() => {
    try {
      const [p, path] = readFileSync(join(ud, 'DevToolsActivePort'), 'utf8').trim().split('\n').map(s => s.trim());
      clearInterval(iv); res(`ws://127.0.0.1:${p}${path}`);
    } catch { if (Date.now() - t0 > 20000) { clearInterval(iv); rej(new Error('Edge nepasileido (nėra DevToolsActivePort)')); } }
  }, 100);
});
const bws = new WebSocket(wsUrl);
bws.addEventListener('error', () => die(1, 'CDP ws klaida'));
bws.addEventListener('close', () => { if (!cleaned) die(1, 'CDP ws uždarytas'); });
await new Promise(r => bws.addEventListener('open', r));
let id = 0; const pend = new Map(); let sess = null;
bws.addEventListener('message', e => {
  const m = JSON.parse(e.data);
  if (m.id && pend.has(m.id)) { const [ok, ko, tm] = pend.get(m.id); clearTimeout(tm); pend.delete(m.id); m.error ? ko(new Error(JSON.stringify(m.error))) : ok(m.result); }
});
const CDP_TIMEOUT_MS = 15000;
const send = (method, params = {}, sessionId = sess) => new Promise((ok, ko) => {
  const i = ++id;
  const tm = setTimeout(() => { pend.delete(i); ko(new Error(`CDP ${method}: laiko riba ${CDP_TIMEOUT_MS} ms`)); }, CDP_TIMEOUT_MS);
  pend.set(i, [ok, ko, tm]);
  bws.send(JSON.stringify({ id: i, method, params, ...(sessionId ? { sessionId } : {}) }));
});
const { targetId } = await send('Target.createTarget', { url: 'about:blank' }, undefined);
sess = (await send('Target.attachToTarget', { targetId, flatten: true }, undefined)).sessionId;
await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send('Emulation.setDefaultBackgroundColorOverride', { color: { r: 0, g: 0, b: 0, a: 0 } });
await send('Page.navigate', { url: `http://127.0.0.1:${port}/host.html` });
let ready = false;
for (let i = 0; i < 100 && !ready; i++) {
  ready = (await send('Runtime.evaluate', { expression: 'window.__ready===true' })).result.value === true;
  if (!ready) await new Promise(r => setTimeout(r, 100));
}
if (!ready) throw new Error('grafika nepasiruošė per 10 s (window.__ready nebuvo true)');

const t0 = Date.now(); let n = 0;
while (Date.now() - t0 < SECONDS * 1000) {
  const { data } = await send('Page.captureScreenshot', { format: 'png', fromSurface: true });
  const f = png(Buffer.from(data, 'base64'));
  if (f.w !== W || f.h !== H) throw new Error(`dydis ${f.w}x${f.h}`);
  if (!process.stdout.write(f.rgba)) await new Promise(r => process.stdout.once('drain', r));
  n++;
}
die(0, `${n} kadrų per ${SECONDS} s (${(n / SECONDS).toFixed(1)} fps), ${W}x${H}`);
