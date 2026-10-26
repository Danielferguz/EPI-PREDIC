// Gráficos SVG de modelos de predicción: calibración, ROC, distribución de riesgos, curva de decisión, validación cruzada de λ y tamaño de muestra.
export const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export const PAL = ['#2E7F90', '#8B2E3B', '#4D4D4D', '#5BA7B8', '#C46B78', '#8FA9AF', '#B08D57', '#9A9A9A'];
export const BLUE = '#2E7F90', RED = '#8B2E3B', GREY = '#8F9699';
export const STYLE = '<style>text{font:12px system-ui,sans-serif;fill:currentColor}.h{font-weight:600}.m{fill:var(--muted,#666)}.s{font-size:11px}</style>';

export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = Math.pow(10, Math.floor(Math.log10(span / n))), m = span / step0 > 5 * n / 1.2 ? 5 : span / step0 > 2 * n / 1.2 ? 2 : 1, st = step0 * m, out = [];
  for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + 1e-9 * span; v += st) out.push(+v.toFixed(10));
  return out;
}
const lin = (lo, hi, a, b) => v => a + (v - lo) / ((hi - lo) || 1) * (b - a);
export const path = (pts, sx, sy) => pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join('');
export function frame({ W = 440, H = 400, xdom, ydom, xlab = '', ylab = '', title = '', sub = '', ml = 56, mb = 46, mr = 16, extraTop = 0, yticks = true }) {
  const m = { l: ml, r: mr, t: (title ? 30 : 14) + (sub ? 16 : 0) + extraTop, b: mb }, sx = lin(xdom[0], xdom[1], m.l, W - m.r), sy = lin(ydom[0], ydom[1], H - m.b, m.t);
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" class="plot" role="img">${STYLE}`;
  if (title) s += `<text x="8" y="18" class="h">${esc(title)}</text>`;
  if (sub) s += `<text x="8" y="34" class="s m">${esc(sub)}</text>`;
  s += `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="none" stroke="currentColor" stroke-width=".7"/>`;
  niceTicks(xdom[0], xdom[1]).filter(v => v >= xdom[0] - 1e-9 && v <= xdom[1] + 1e-9).forEach(v => (s += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${H - m.b}" y2="${H - m.b + 4}" stroke="currentColor"/><text x="${sx(v)}" y="${H - m.b + 17}" text-anchor="middle" class="s">${+v.toPrecision(6)}</text>`));
  if (yticks) niceTicks(ydom[0], ydom[1]).filter(v => v >= ydom[0] - 1e-9 && v <= ydom[1] + 1e-9).forEach(v => (s += `<line x1="${m.l - 4}" x2="${m.l}" y1="${sy(v)}" y2="${sy(v)}" stroke="currentColor"/><text x="${m.l - 7}" y="${sy(v) + 4}" text-anchor="end" class="s">${+v.toPrecision(6)}</text>`));
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 8}" text-anchor="middle">${esc(xlab)}</text><text transform="translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${esc(ylab)}</text>`;
  return { s, sx, sy, m, W, H };
}
const histBins = (v, lo, hi, k) => { const c = new Array(k).fill(0); for (const x of v) { let j = Math.floor((x - lo) / (hi - lo) * k); j = Math.min(k - 1, Math.max(0, j)); c[j]++; } return c; };

// comparación de modelos: rendimiento aparente (círculo hueco) y corregido por optimismo (círculo lleno), un modelo por fila
export function comparePlot({ rows, xlab = 'C statistic', title = 'Model comparison (optimism-corrected)' }) {
  const vals = rows.flatMap(r => [r.apparent, r.corrected]).filter(isFinite), lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.15 + 0.005, best = Math.max(...rows.map(r => r.corrected));
  const H = 70 + rows.length * 34, F = frame({ W: 640, H, xdom: [lo - pad, hi + pad], ydom: [0, rows.length], xlab, title, ml: 230, mb: 44, yticks: false }); let s = F.s;
  rows.forEach((r, i) => {
    const y = F.sy(i + 0.5);
    s += `<line x1="${F.m.l}" x2="${F.W - F.m.r}" y1="${y}" y2="${y}" stroke="currentColor" stroke-opacity=".08"/><text x="${F.m.l - 8}" y="${y + 4}" text-anchor="end" class="s">${esc(r.label.length > 38 ? r.label.slice(0, 37) + '…' : r.label)}</text>`;
    s += `<line x1="${F.sx(r.corrected)}" x2="${F.sx(r.apparent)}" y1="${y}" y2="${y}" stroke="${GREY}" stroke-width="1.4"/><circle cx="${F.sx(r.apparent)}" cy="${y}" r="4.4" fill="#fff" stroke="${GREY}" stroke-width="1.6"/><circle cx="${F.sx(r.corrected)}" cy="${y}" r="5.4" fill="${r.corrected === best ? BLUE : RED}" stroke="#fff"/>`;
  });
  [[GREY, 'Apparent', false], [RED, 'Optimism-corrected', true]].forEach(([c, t, f], q) => (s += `<circle cx="${F.W - 300 + q * 100}" cy="14" r="4.4" fill="${f ? c : '#fff'}" stroke="${c}" stroke-width="1.6"/><text x="${F.W - 291 + q * 100}" y="18" class="s">${t}</text>`));
  return s + '</svg>';
}

// calibración: curva suavizada (lowess), grupos con IC y distribución de las predicciones (por desenlace en binarios)
export function calibrationPlot({ groups, curve, pred, y = null, title = 'Calibration', xlab = 'Predicted risk', ylab = 'Observed risk', note = '', max = null }) {
  const mx = max ?? Math.min(1, Math.max(0.05, ...groups.map(g => Math.max(g.pred, g.obs, g.hi || 0)), ...(curve || []).map(p => Math.max(p[0], p[1]))) * 1.08), hb = 54;
  const F = frame({ W: 460, H: 470, xdom: [0, mx], ydom: [0, mx], xlab, ylab, title, sub: note, mb: 46 + hb }); let s = F.s; const { sx, sy, m } = F;
  s += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(mx)}" y2="${sy(mx)}" stroke="currentColor" stroke-dasharray="4 3"/>`;
  if (curve?.length) s += `<path d="${path(curve.filter(p => p[1] >= -0.02 && p[1] <= mx * 1.1), sx, sy)}" fill="none" stroke="${RED}" stroke-width="2"/>`;
  groups.forEach(g => { if (g.lo != null) s += `<line x1="${sx(g.pred)}" x2="${sx(g.pred)}" y1="${sy(Math.max(0, g.lo))}" y2="${sy(Math.min(mx, g.hi))}" stroke="${BLUE}"/>`; s += `<circle cx="${sx(g.pred)}" cy="${sy(g.obs)}" r="4.5" fill="${BLUE}"/>`; });
  // distribución de las predicciones (histograma, por desenlace si es binario)
  const k = 30, y0 = F.H - 46 - hb + 10, hmax = Math.max(1, ...histBins(pred, 0, mx, k));
  const draw = (vals, col, off) => histBins(vals, 0, mx, k).forEach((c, j) => { const h = c / hmax * (hb - 14), x0 = sx(mx * j / k), w = (sx(mx * (j + 1) / k) - x0) - 1; s += `<rect x="${x0 + off}" y="${y0 + hb - 14 - h}" width="${Math.max(0.5, w / (y ? 2 : 1))}" height="${h}" fill="${col}" opacity=".85"/>`; });
  if (y) { draw(pred.filter((_, i) => !y[i]), GREY, 0); draw(pred.filter((_, i) => y[i]), RED, (sx(mx / k) - sx(0)) / 2); s += `<text x="${m.l + 4}" y="${y0 + hb - 1}" class="s m">Distribution of predicted risks: <tspan fill="${GREY}">no event</tspan> / <tspan fill="${RED}">event</tspan></text>`; }
  else { draw(Array.from(pred), GREY, 0); s += `<text x="${m.l + 4}" y="${y0 + hb - 1}" class="s m">Distribution of predicted risks</text>`; }
  return s + '</svg>';
}
export function scatterCalib({ pred, y, curve, title = 'Calibration', xlab = 'Predicted value', ylab = 'Observed value', note = '' }) {
  const all = [...pred, ...y], lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.04, F = frame({ W: 460, H: 440, xdom: [lo - pad, hi + pad], ydom: [lo - pad, hi + pad], xlab, ylab, title, sub: note }); let s = F.s;
  s += `<line x1="${F.sx(lo - pad)}" y1="${F.sy(lo - pad)}" x2="${F.sx(hi + pad)}" y2="${F.sy(hi + pad)}" stroke="currentColor" stroke-dasharray="4 3"/>`;
  const step = Math.max(1, Math.floor(pred.length / 1500)); for (let i = 0; i < pred.length; i += step) s += `<circle cx="${F.sx(pred[i])}" cy="${F.sy(y[i])}" r="2.2" fill="${BLUE}" fill-opacity=".35"/>`;
  if (curve?.length) s += `<path d="${path(curve, F.sx, F.sy)}" fill="none" stroke="${RED}" stroke-width="2"/>`;
  return s + '</svg>';
}
export function rocPlot(pts, aucTxt, title = 'ROC curve') {
  const F = frame({ W: 420, H: 400, xdom: [0, 1], ydom: [0, 1], xlab: '1 − Specificity', ylab: 'Sensitivity', title: `${title} (${aucTxt})` }); let s = F.s;
  s += `<line x1="${F.sx(0)}" y1="${F.sy(0)}" x2="${F.sx(1)}" y2="${F.sy(1)}" stroke="currentColor" stroke-dasharray="4 3"/>`;
  return s + `<path d="${path(pts, F.sx, F.sy)}" fill="none" stroke="${BLUE}" stroke-width="2"/></svg>`;
}
export function rocPoints(score, y) {
  const idx = Array.from(score.keys()).sort((a, b) => score[b] - score[a]), n1 = y.reduce((a, b) => a + b, 0), n0 = y.length - n1, pts = [[0, 0]]; let tp = 0, fp = 0;
  for (let k = 0; k < idx.length; k++) { if (y[idx[k]]) tp++; else fp++; if (k === idx.length - 1 || score[idx[k + 1]] !== score[idx[k]]) pts.push([fp / n0, tp / n1]); }
  return pts;
}
export function dcaPlot(rows, { title = 'Decision curve', ymax = null } = {}) {
  const ymx = ymax ?? Math.max(0.05, ...rows.flatMap(r => [r.model, r.all])) * 1.1, ymn = -Math.min(0.1, ymx * 0.5), F = frame({ W: 520, H: 380, xdom: [rows[0].pt, rows[rows.length - 1].pt], ydom: [ymn, ymx], xlab: 'Risk threshold (pt)', ylab: 'Net benefit', title, ml: 62 }); let s = F.s;
  const clip = v => Math.max(ymn, v);
  s += `<line x1="${F.m.l}" x2="${F.W - F.m.r}" y1="${F.sy(0)}" y2="${F.sy(0)}" stroke="${GREY}" stroke-width="1.6"/>`;
  s += `<path d="${path(rows.map(r => [r.pt, clip(r.all)]), F.sx, F.sy)}" fill="none" stroke="${GREY}" stroke-width="1.6" stroke-dasharray="5 3"/>`;
  s += `<path d="${path(rows.map(r => [r.pt, clip(r.model)]), F.sx, F.sy)}" fill="none" stroke="${BLUE}" stroke-width="2.4"/>`;
  const lx = F.W - 200, ly = F.m.t + 10;
  [[BLUE, 'Model', ''], [GREY, 'Treat all', '5 3'], [GREY, 'Treat none', '']].forEach(([c, t, d], k) => (s += `<line x1="${lx}" x2="${lx + 24}" y1="${ly + k * 16}" y2="${ly + k * 16}" stroke="${c}" stroke-width="2" ${d ? `stroke-dasharray="${d}"` : ''}/><text x="${lx + 30}" y="${ly + 4 + k * 16}" class="s">${t}</text>`));
  return s + '</svg>';
}
export function cvPlot({ x, y, se, xmark, xmark2, title = 'Cross-validation for λ', xlab = 'log(λ)', ylab = 'Cross-validated deviance' }) {
  const lo = Math.min(...y.map((v, i) => v - (se?.[i] || 0))), hi = Math.max(...y.map((v, i) => v + (se?.[i] || 0))), F = frame({ W: 480, H: 340, xdom: [Math.min(...x), Math.max(...x)], ydom: [lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.05], xlab, ylab, title, ml: 62 }); let s = F.s;
  if (se) y.forEach((v, i) => (s += `<line x1="${F.sx(x[i])}" x2="${F.sx(x[i])}" y1="${F.sy(v - se[i])}" y2="${F.sy(v + se[i])}" stroke="${GREY}"/>`));
  s += `<path d="${path(x.map((v, i) => [v, y[i]]), F.sx, F.sy)}" fill="none" stroke="${BLUE}" stroke-width="1.6"/>`;
  x.forEach((v, i) => (s += `<circle cx="${F.sx(v)}" cy="${F.sy(y[i])}" r="2.6" fill="${RED}"/>`));
  for (const [xm, col] of [[xmark, BLUE], [xmark2, GREY]]) if (xm != null) s += `<line x1="${F.sx(xm)}" x2="${F.sx(xm)}" y1="${F.m.t}" y2="${F.H - F.m.b}" stroke="${col}" stroke-dasharray="4 3"/>`;
  return s + '</svg>';
}
// tamaño de muestra mínimo según R²_CS para distintos números de parámetros
export function nPlot(curves, { title = 'Minimum sample size by Cox-Snell R²', xlab = 'Expected Cox-Snell R²', mark = null }) {
  const xs = curves.flatMap(c => c.pts.map(p => p[0])), ys = curves.flatMap(c => c.pts.map(p => p[1])), F = frame({ W: 560, H: 360, xdom: [Math.min(...xs), Math.max(...xs)], ydom: [0, Math.max(...ys) * 1.05], xlab, ylab: 'Minimum n', title, ml: 66 }); let s = F.s;
  curves.forEach((c, k) => { const col = PAL[k % PAL.length]; s += `<path d="${path(c.pts, F.sx, F.sy)}" fill="none" stroke="${col}" stroke-width="2"/><rect x="${F.W - 130}" y="${F.m.t + 8 + k * 16}" width="10" height="10" fill="${col}"/><text x="${F.W - 115}" y="${F.m.t + 17 + k * 16}" class="s">${esc(c.label)}</text>`; });
  if (mark) s += `<circle cx="${F.sx(mark[0])}" cy="${F.sy(mark[1])}" r="5" fill="none" stroke="${RED}" stroke-width="2"/>`;
  return s + '</svg>';
}

// ---------------------------------------------------------------- gráficos adicionales de rendimiento (texto en inglés para publicación)
// distribución de los riesgos predichos según el desenlace (binario): cuanto más separadas, mejor discrimina
export function riskDistPlot({ pred, y, title = 'Distribution of predicted risk by outcome' }) {
  const mx = Math.min(1, Math.max(0.05, Math.max(...pred) * 1.05)), k = 40, ev = pred.filter((_, i) => y[i]), ne = pred.filter((_, i) => !y[i]);
  const dens = v => histBins(v, 0, mx, k).map(c => c / (v.length || 1) / (mx / k)), d1 = dens(ev), d0 = dens(ne), ym = Math.max(...d1, ...d0) * 1.08;
  const m1 = ev.reduce((a, b) => a + b, 0) / (ev.length || 1), m0 = ne.reduce((a, b) => a + b, 0) / (ne.length || 1);
  const F = frame({ W: 500, H: 360, xdom: [0, mx], ydom: [0, ym], xlab: 'Predicted risk', ylab: 'Density', title, sub: `Discrimination slope = ${(m1 - m0).toFixed(3)} (mean risk: events ${m1.toFixed(3)}, non-events ${m0.toFixed(3)})` }); let s = F.s;
  [[d0, GREY, 'No event'], [d1, RED, 'Event']].forEach(([d, col], q) => { s += `<path d="${path([[0, 0], ...d.flatMap((v, j) => [[mx * j / k, v], [mx * (j + 1) / k, v]]), [mx, 0]], F.sx, F.sy)}Z" fill="${col}" fill-opacity=".35" stroke="${col}" stroke-width="1.4"/>`; });
  [[m0, GREY], [m1, RED]].forEach(([m, col]) => (s += `<line x1="${F.sx(m)}" x2="${F.sx(m)}" y1="${F.m.t}" y2="${F.H - F.m.b}" stroke="${col}" stroke-dasharray="4 3"/>`));
  [[GREY, 'No event'], [RED, 'Event']].forEach(([c, t], q) => (s += `<rect x="${F.W - 110}" y="${F.m.t + 8 + q * 16}" width="10" height="10" fill="${c}" opacity=".6"/><text x="${F.W - 95}" y="${F.m.t + 17 + q * 16}" class="s">${t}</text>`));
  return s + '</svg>';
}
// sensibilidad, especificidad, VPP y VPN según el umbral de riesgo
export function thresholdPlot({ pred, y, title = 'Classification performance by risk threshold', marks = [] }) {
  const n = y.length, P = y.reduce((a, b) => a + b, 0), N = n - P, mx = Math.min(0.99, Math.max(0.1, Math.max(...pred))), ths = Array.from({ length: 60 }, (_, i) => 0.005 + (mx - 0.005) * i / 59), rows = ths.map(t => { let tp = 0, fp = 0; for (let i = 0; i < n; i++) if (pred[i] >= t) { if (y[i]) tp++; else fp++; } const pos = tp + fp; return { t, se: tp / P, sp: 1 - fp / N, ppv: pos ? tp / pos : NaN, npv: (N - fp) / ((N - fp) + (P - tp)) }; });
  const F = frame({ W: 540, H: 380, xdom: [0, mx], ydom: [0, 1], xlab: 'Risk threshold', ylab: 'Proportion', title, extraTop: 18 }); let s = F.s;
  const L = [['se', 'Sensitivity', BLUE, ''], ['sp', 'Specificity', RED, ''], ['ppv', 'PPV', BLUE, '5 3'], ['npv', 'NPV', RED, '5 3']];
  L.forEach(([k, , col, dash]) => (s += `<path d="${path(rows.filter(r => isFinite(r[k])).map(r => [r.t, r[k]]), F.sx, F.sy)}" fill="none" stroke="${col}" stroke-width="2" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`));
  marks.forEach(t => (s += `<line x1="${F.sx(t)}" x2="${F.sx(t)}" y1="${F.m.t}" y2="${F.H - F.m.b}" stroke="${GREY}" stroke-dasharray="2 3"/>`));
  L.forEach(([, t, col, dash], q) => (s += `<line x1="${70 + q * 110}" x2="${94 + q * 110}" y1="36" y2="36" stroke="${col}" stroke-width="2" ${dash ? `stroke-dasharray="${dash}"` : ''}/><text x="${100 + q * 110}" y="40" class="s">${t}</text>`));
  return s + '</svg>';
}
// riesgo observado por grupos de riesgo predicho (barras con IC 95 %) y riesgo medio predicho (rombos)
export function decilePlot({ groups, title = 'Observed vs predicted risk by risk group' }) {
  const mx = Math.min(1, Math.max(0.05, ...groups.map(g => Math.max(g.hi ?? g.obs, g.pred))) * 1.1), n = groups.length, F = frame({ W: 540, H: 380, xdom: [0.5, n + 0.5], ydom: [0, mx], xlab: 'Risk group (low → high predicted risk)', ylab: 'Risk', title }); let s = F.s;
  const w = (F.sx(1.5) - F.sx(0.5)) * 0.6;
  groups.forEach((g, i) => { const x = F.sx(i + 1); s += `<rect x="${x - w / 2}" y="${F.sy(g.obs)}" width="${w}" height="${F.sy(0) - F.sy(g.obs)}" fill="${BLUE}" fill-opacity=".55"/>`; if (g.lo != null) s += `<line x1="${x}" x2="${x}" y1="${F.sy(Math.max(0, g.lo))}" y2="${F.sy(Math.min(mx, g.hi))}" stroke="#222" stroke-width="1.2"/>`; s += `<polygon points="${x},${F.sy(g.pred) - 6} ${x + 6},${F.sy(g.pred)} ${x},${F.sy(g.pred) + 6} ${x - 6},${F.sy(g.pred)}" fill="${RED}" stroke="#fff" stroke-width=".8"/><text x="${x}" y="${F.H - F.m.b + 17}" text-anchor="middle" class="s">${i + 1}</text>`; });
  s += `<rect x="${F.W - 170}" y="${F.m.t + 6}" width="10" height="10" fill="${BLUE}" opacity=".6"/><text x="${F.W - 155}" y="${F.m.t + 15}" class="s">Observed (95% CI)</text><polygon points="${F.W - 165},${F.m.t + 22} ${F.W - 159},${F.m.t + 28} ${F.W - 165},${F.m.t + 34} ${F.W - 171},${F.m.t + 28}" fill="${RED}"/><text x="${F.W - 155}" y="${F.m.t + 32}" class="s">Mean predicted</text>`;
  return s + '</svg>';
}
// barras horizontales (importancia de los predictores); items: [{label, value, mark?}]
export function importancePlot({ items, title = 'Predictor importance', xlab = 'Partial Wald χ² − df' }) {
  const rowH = 24, H = 70 + items.length * rowH, ml = Math.min(260, 7 * Math.max(...items.map(i => i.label.length), 8) + 20), mx = Math.max(...items.map(i => i.value), 1) * 1.08, F = frame({ W: 640, H, xdom: [0, mx], ydom: [0, items.length], xlab, ylab: '', title, ml, mb: 44, yticks: false }); let s = F.s;
  const rh = (F.H - F.m.t - F.m.b) / items.length;
  items.forEach((it, i) => { const h = rh * 0.62, yy = F.m.t + i * rh + rh * 0.19, x1 = F.sx(Math.max(0, it.value)); s += `<rect x="${F.sx(0)}" y="${yy}" width="${Math.max(0, x1 - F.sx(0))}" height="${h}" fill="${it.keep === false ? GREY : BLUE}"/><text x="${ml - 6}" y="${yy + h / 2 + 4}" text-anchor="end" class="s">${esc(it.label)}</text><text x="${x1 + 4}" y="${yy + h / 2 + 4}" class="s">${(+it.value).toFixed(1)}</text>`; });
  return s + '</svg>';
}
// rendimiento (aparente y corregido por optimismo) según el número de predictores; kStar marcado
export function pathPlot({ rows, kStar, tol, ylab = 'C statistic', title = 'Performance by number of predictors (optimism-corrected)' }) {
  const ys = rows.flatMap(r => [r.apparent, r.corrected]), lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.12 + 0.005, full = rows[rows.length - 1].corrected, F = frame({ W: 560, H: 380, xdom: [1, rows.length], ydom: [lo - pad, hi + pad], xlab: 'Number of predictors (added by importance)', ylab, title, ml: 64 }); let s = F.s;
  s += `<rect x="${F.m.l}" y="${F.sy(full)}" width="${F.W - F.m.l - F.m.r}" height="${F.sy(full - tol) - F.sy(full)}" fill="${BLUE}" fill-opacity=".10"/><line x1="${F.m.l}" x2="${F.W - F.m.r}" y1="${F.sy(full - tol)}" y2="${F.sy(full - tol)}" stroke="${BLUE}" stroke-dasharray="4 3"/>`;
  s += `<path d="${path(rows.map(r => [r.k, r.apparent]), F.sx, F.sy)}" fill="none" stroke="${GREY}" stroke-width="1.6" stroke-dasharray="5 3"/><path d="${path(rows.map(r => [r.k, r.corrected]), F.sx, F.sy)}" fill="none" stroke="${RED}" stroke-width="2.4"/>`;
  rows.forEach(r => (s += `<circle cx="${F.sx(r.k)}" cy="${F.sy(r.corrected)}" r="${r.k === kStar ? 6 : 3.4}" fill="${r.k === kStar ? BLUE : RED}" stroke="#fff"/>`));
  if (kStar) s += `<text x="${F.sx(kStar)}" y="${F.sy(rows[kStar - 1].corrected) - 12}" text-anchor="middle" class="s h">k = ${kStar}</text>`;
  [[RED, 'Optimism-corrected', ''], [GREY, 'Apparent', '5 3']].forEach(([c, t, d], q) => (s += `<line x1="${F.W - 190}" x2="${F.W - 166}" y1="${F.m.t + 12 + q * 16}" y2="${F.m.t + 12 + q * 16}" stroke="${c}" stroke-width="2" ${d ? `stroke-dasharray="${d}"` : ''}/><text x="${F.W - 160}" y="${F.m.t + 16 + q * 16}" class="s">${t}</text>`));
  return s + `<text x="${F.m.l + 6}" y="${F.sy(full - tol) - 4}" class="s m">full model − ${tol}</text></svg>`;
}
