// Gráficos SVG de modelos de predicción: calibración, ROC, distribución de riesgos, curva de decisión, validación cruzada de λ y tamaño de muestra.
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export const PAL = ['#2E7F90', '#8B2E3B', '#4D4D4D', '#5BA7B8', '#C46B78', '#8FA9AF', '#B08D57', '#9A9A9A'];
const BLUE = '#2E7F90', RED = '#8B2E3B', GREY = '#8F9699';
const STYLE = '<style>text{font:12px system-ui,sans-serif;fill:currentColor}.h{font-weight:600}.m{fill:var(--muted,#666)}.s{font-size:11px}</style>';

export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = Math.pow(10, Math.floor(Math.log10(span / n))), m = span / step0 > 5 * n / 1.2 ? 5 : span / step0 > 2 * n / 1.2 ? 2 : 1, st = step0 * m, out = [];
  for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + 1e-9 * span; v += st) out.push(+v.toFixed(10));
  return out;
}
const lin = (lo, hi, a, b) => v => a + (v - lo) / ((hi - lo) || 1) * (b - a);
const path = (pts, sx, sy) => pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('');
function frame({ W = 440, H = 400, xdom, ydom, xlab = '', ylab = '', title = '', sub = '', ml = 56, mb = 46, mr = 16, extraTop = 0 }) {
  const m = { l: ml, r: mr, t: (title ? 30 : 14) + (sub ? 16 : 0) + extraTop, b: mb }, sx = lin(xdom[0], xdom[1], m.l, W - m.r), sy = lin(ydom[0], ydom[1], H - m.b, m.t);
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" class="plot" role="img">${STYLE}`;
  if (title) s += `<text x="8" y="18" class="h">${esc(title)}</text>`;
  if (sub) s += `<text x="8" y="34" class="s m">${esc(sub)}</text>`;
  s += `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="none" stroke="currentColor" stroke-width=".7"/>`;
  niceTicks(xdom[0], xdom[1]).filter(v => v >= xdom[0] - 1e-9 && v <= xdom[1] + 1e-9).forEach(v => (s += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${H - m.b}" y2="${H - m.b + 4}" stroke="currentColor"/><text x="${sx(v)}" y="${H - m.b + 17}" text-anchor="middle" class="s">${+v.toPrecision(6)}</text>`));
  niceTicks(ydom[0], ydom[1]).filter(v => v >= ydom[0] - 1e-9 && v <= ydom[1] + 1e-9).forEach(v => (s += `<line x1="${m.l - 4}" x2="${m.l}" y1="${sy(v)}" y2="${sy(v)}" stroke="currentColor"/><text x="${m.l - 7}" y="${sy(v) + 4}" text-anchor="end" class="s">${+v.toPrecision(6)}</text>`));
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 8}" text-anchor="middle">${esc(xlab)}</text><text transform="translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${esc(ylab)}</text>`;
  return { s, sx, sy, m, W, H };
}
const histBins = (v, lo, hi, k) => { const c = new Array(k).fill(0); for (const x of v) { let j = Math.floor((x - lo) / (hi - lo) * k); j = Math.min(k - 1, Math.max(0, j)); c[j]++; } return c; };

// calibración: curva suavizada (lowess), grupos con IC y distribución de las predicciones (por desenlace en binarios)
export function calibrationPlot({ groups, curve, pred, y = null, title = 'Calibración', xlab = 'Riesgo predicho', ylab = 'Riesgo observado', note = '', max = null }) {
  const mx = max ?? Math.min(1, Math.max(0.05, ...groups.map(g => Math.max(g.pred, g.obs, g.hi || 0)), ...(curve || []).map(p => Math.max(p[0], p[1]))) * 1.08), hb = 54;
  const F = frame({ W: 460, H: 470, xdom: [0, mx], ydom: [0, mx], xlab, ylab, title, sub: note, mb: 46 + hb }); let s = F.s; const { sx, sy, m } = F;
  s += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(mx)}" y2="${sy(mx)}" stroke="currentColor" stroke-dasharray="4 3"/>`;
  if (curve?.length) s += `<path d="${path(curve.filter(p => p[1] >= -0.02 && p[1] <= mx * 1.1), sx, sy)}" fill="none" stroke="${RED}" stroke-width="2"/>`;
  groups.forEach(g => { if (g.lo != null) s += `<line x1="${sx(g.pred)}" x2="${sx(g.pred)}" y1="${sy(Math.max(0, g.lo))}" y2="${sy(Math.min(mx, g.hi))}" stroke="${BLUE}"/>`; s += `<circle cx="${sx(g.pred)}" cy="${sy(g.obs)}" r="4.5" fill="${BLUE}"/>`; });
  // distribución de las predicciones (histograma, por desenlace si es binario)
  const k = 30, y0 = F.H - 46 - hb + 10, hmax = Math.max(1, ...histBins(pred, 0, mx, k));
  const draw = (vals, col, off) => histBins(vals, 0, mx, k).forEach((c, j) => { const h = c / hmax * (hb - 14), x0 = sx(mx * j / k), w = (sx(mx * (j + 1) / k) - x0) - 1; s += `<rect x="${x0 + off}" y="${y0 + hb - 14 - h}" width="${Math.max(0.5, w / (y ? 2 : 1))}" height="${h}" fill="${col}" opacity=".85"/>`; });
  if (y) { draw(pred.filter((_, i) => !y[i]), GREY, 0); draw(pred.filter((_, i) => y[i]), RED, (sx(mx / k) - sx(0)) / 2); s += `<text x="${m.l + 4}" y="${y0 + hb - 1}" class="s m">Distribución de riesgos: <tspan fill="${GREY}">sin evento</tspan> / <tspan fill="${RED}">evento</tspan></text>`; }
  else { draw(Array.from(pred), GREY, 0); s += `<text x="${m.l + 4}" y="${y0 + hb - 1}" class="s m">Distribución de los riesgos predichos</text>`; }
  return s + '</svg>';
}
export function scatterCalib({ pred, y, curve, title = 'Calibración', xlab = 'Valor predicho', ylab = 'Valor observado', note = '' }) {
  const all = [...pred, ...y], lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.04, F = frame({ W: 460, H: 440, xdom: [lo - pad, hi + pad], ydom: [lo - pad, hi + pad], xlab, ylab, title, sub: note }); let s = F.s;
  s += `<line x1="${F.sx(lo - pad)}" y1="${F.sy(lo - pad)}" x2="${F.sx(hi + pad)}" y2="${F.sy(hi + pad)}" stroke="currentColor" stroke-dasharray="4 3"/>`;
  const step = Math.max(1, Math.floor(pred.length / 1500)); for (let i = 0; i < pred.length; i += step) s += `<circle cx="${F.sx(pred[i])}" cy="${F.sy(y[i])}" r="2.2" fill="${BLUE}" fill-opacity=".35"/>`;
  if (curve?.length) s += `<path d="${path(curve, F.sx, F.sy)}" fill="none" stroke="${RED}" stroke-width="2"/>`;
  return s + '</svg>';
}
export function rocPlot(pts, aucTxt, title = 'Curva ROC') {
  const F = frame({ W: 420, H: 400, xdom: [0, 1], ydom: [0, 1], xlab: '1 − Especificidad', ylab: 'Sensibilidad', title: `${title} (${aucTxt})` }); let s = F.s;
  s += `<line x1="${F.sx(0)}" y1="${F.sy(0)}" x2="${F.sx(1)}" y2="${F.sy(1)}" stroke="currentColor" stroke-dasharray="4 3"/>`;
  return s + `<path d="${path(pts, F.sx, F.sy)}" fill="none" stroke="${BLUE}" stroke-width="2"/></svg>`;
}
export function rocPoints(score, y) {
  const idx = Array.from(score.keys()).sort((a, b) => score[b] - score[a]), n1 = y.reduce((a, b) => a + b, 0), n0 = y.length - n1, pts = [[0, 0]]; let tp = 0, fp = 0;
  for (let k = 0; k < idx.length; k++) { if (y[idx[k]]) tp++; else fp++; if (k === idx.length - 1 || score[idx[k + 1]] !== score[idx[k]]) pts.push([fp / n0, tp / n1]); }
  return pts;
}
export function dcaPlot(rows, { title = 'Curva de decisión', ymax = null } = {}) {
  const ymx = ymax ?? Math.max(0.05, ...rows.flatMap(r => [r.model, r.all])) * 1.1, ymn = -Math.min(0.1, ymx * 0.5), F = frame({ W: 520, H: 380, xdom: [rows[0].pt, rows[rows.length - 1].pt], ydom: [ymn, ymx], xlab: 'Umbral de probabilidad (pt)', ylab: 'Beneficio neto', title, ml: 62 }); let s = F.s;
  const clip = v => Math.max(ymn, v);
  s += `<line x1="${F.m.l}" x2="${F.W - F.m.r}" y1="${F.sy(0)}" y2="${F.sy(0)}" stroke="${GREY}" stroke-width="1.6"/>`;
  s += `<path d="${path(rows.map(r => [r.pt, clip(r.all)]), F.sx, F.sy)}" fill="none" stroke="${GREY}" stroke-width="1.6" stroke-dasharray="5 3"/>`;
  s += `<path d="${path(rows.map(r => [r.pt, clip(r.model)]), F.sx, F.sy)}" fill="none" stroke="${BLUE}" stroke-width="2.4"/>`;
  const lx = F.W - 200, ly = F.m.t + 10;
  [[BLUE, 'Modelo', ''], [GREY, 'Tratar a todos', '5 3'], [GREY, 'No tratar a nadie', '']].forEach(([c, t, d], k) => (s += `<line x1="${lx}" x2="${lx + 24}" y1="${ly + k * 16}" y2="${ly + k * 16}" stroke="${c}" stroke-width="2" ${d ? `stroke-dasharray="${d}"` : ''}/><text x="${lx + 30}" y="${ly + 4 + k * 16}" class="s">${t}</text>`));
  return s + '</svg>';
}
export function cvPlot({ x, y, se, xmark, xmark2, title = 'Validación cruzada de λ', xlab = 'log(λ)', ylab = 'Devianza de validación cruzada' }) {
  const lo = Math.min(...y.map((v, i) => v - (se?.[i] || 0))), hi = Math.max(...y.map((v, i) => v + (se?.[i] || 0))), F = frame({ W: 480, H: 340, xdom: [Math.min(...x), Math.max(...x)], ydom: [lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.05], xlab, ylab, title, ml: 62 }); let s = F.s;
  if (se) y.forEach((v, i) => (s += `<line x1="${F.sx(x[i])}" x2="${F.sx(x[i])}" y1="${F.sy(v - se[i])}" y2="${F.sy(v + se[i])}" stroke="${GREY}"/>`));
  s += `<path d="${path(x.map((v, i) => [v, y[i]]), F.sx, F.sy)}" fill="none" stroke="${BLUE}" stroke-width="1.6"/>`;
  x.forEach((v, i) => (s += `<circle cx="${F.sx(v)}" cy="${F.sy(y[i])}" r="2.6" fill="${RED}"/>`));
  for (const [xm, col] of [[xmark, BLUE], [xmark2, GREY]]) if (xm != null) s += `<line x1="${F.sx(xm)}" x2="${F.sx(xm)}" y1="${F.m.t}" y2="${F.H - F.m.b}" stroke="${col}" stroke-dasharray="4 3"/>`;
  return s + '</svg>';
}
// tamaño de muestra mínimo según R²_CS para distintos números de parámetros
export function nPlot(curves, { title = 'Tamaño de muestra mínimo según el R² de Cox-Snell', xlab = 'R² de Cox-Snell esperado', mark = null }) {
  const xs = curves.flatMap(c => c.pts.map(p => p[0])), ys = curves.flatMap(c => c.pts.map(p => p[1])), F = frame({ W: 560, H: 360, xdom: [Math.min(...xs), Math.max(...xs)], ydom: [0, Math.max(...ys) * 1.05], xlab, ylab: 'n mínimo', title, ml: 66 }); let s = F.s;
  curves.forEach((c, k) => { const col = PAL[k % PAL.length]; s += `<path d="${path(c.pts, F.sx, F.sy)}" fill="none" stroke="${col}" stroke-width="2"/><rect x="${F.W - 130}" y="${F.m.t + 8 + k * 16}" width="10" height="10" fill="${col}"/><text x="${F.W - 115}" y="${F.m.t + 17 + k * 16}" class="s">${esc(c.label)}</text>`; });
  if (mark) s += `<circle cx="${F.sx(mark[0])}" cy="${F.sy(mark[1])}" r="5" fill="none" stroke="${RED}" stroke-width="2"/>`;
  return s + '</svg>';
}
