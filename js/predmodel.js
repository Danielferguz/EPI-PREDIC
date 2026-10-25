// Desarrollo y evaluación de modelos de predicción: diseño con splines cúbicos restringidos, ajuste (logística, Cox, lineal; sin penalizar,
// ridge o lasso), medidas de discriminación y calibración, validación interna por bootstrap con corrección del optimismo (Harrell),
// contracción uniforme, recalibración, calibración por grupos y análisis de curva de decisión.
import { fitGLM, fitCox, fitOLS, coxLogLik, km } from './models.js?v=2026.10.25-1';
import { enet, cvEnet, standardize } from './glmnet.js?v=2026.10.25-1';
import { rng, qnorm, pnorm } from './dist.js?v=2026.10.25-1';
import { isMiss, levelsOf } from './data.js?v=2026.10.25-1';

const expit = e => 1 / (1 + Math.exp(-e)), logit = p => Math.log(p / (1 - p)), sum = a => { let s = 0; for (const v of a) s += v; return s; };
const clampP = p => Math.min(Math.max(p, 1e-9), 1 - 1e-9);
export const tick = () => new Promise(r => setTimeout(r, 0));

// variables transformadas (logaritmo) se materializan como columnas nuevas, de modo que el diseño las trate como cuantitativas
export function materialize(spec, ds, meta) {
  const extra = {}, m2 = { ...meta }, out = { ...spec, vars: spec.vars.map(v => ({ ...v })) };
  for (const v of out.vars) if (v.form === 'log') { const nm = `ln(${v.name})`, col = Float64Array.from(ds.cols[v.name], x => (x > 0 ? Math.log(x) : NaN)); extra[nm] = col; m2[nm] = { name: nm, type: 'num', levels: [], ref: null }; v.name = nm; v.form = 'lin'; }
  return { spec: out, ds: Object.keys(extra).length ? { n: ds.n, cols: { ...ds.cols, ...extra } } : ds, meta: m2 };
}

// ---------------------------------------------------------------- splines cúbicos restringidos (Harrell)
const KPOS = { 3: [0.10, 0.50, 0.90], 4: [0.05, 0.35, 0.65, 0.95], 5: [0.05, 0.275, 0.5, 0.725, 0.95] };
export function quantile(sorted, q) { const h = (sorted.length - 1) * q, lo = Math.floor(h), hi = Math.ceil(h); return sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo); }
export function knotsFor(x, k) { const s = Array.from(x).filter(isFinite).sort((a, b) => a - b); return KPOS[k].map(q => quantile(s, q)); }
export function rcsRow(x, t) { // [x, h1 … h_{k−2}]
  const k = t.length, out = [x], d = Math.pow(t[k - 1] - t[0], 2), pos = v => (v > 0 ? v ** 3 : 0);
  for (let j = 0; j < k - 2; j++) out.push((pos(x - t[j]) - pos(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2]) + pos(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])) / d);
  return out;
}

// ---------------------------------------------------------------- diseño: información (nudos, niveles) y matriz X
export function designInfo(ds, meta, spec, rows) {
  const vars = spec.vars.map(v => {
    const m = meta[v.name];
    if (m.type === 'num') { const col = rows.map(i => ds.cols[v.name][i]), form = /^rcs\d$/.test(v.form || '') ? +v.form.slice(3) : 0, uniq = new Set(col).size; return { name: v.name, kind: 'num', k: form && uniq > form + 1 ? form : 0, knots: form && uniq > form + 1 ? knotsFor(col, form) : null }; }
    return { name: v.name, kind: 'cat', ref: m.ref, levels: levelsOf(m) };
  });
  return { vars };
}
export function columnNames(info) {
  const o = [];
  for (const v of info.vars) { if (v.kind === 'cat') v.levels.slice(1).forEach(l => o.push({ name: `${v.name}: ${l}`, v: v.name, level: l })); else if (v.knots) { o.push({ name: v.name, v: v.name }); for (let j = 0; j < v.k - 2; j++) o.push({ name: `${v.name} (spline ${j + 1})`, v: v.name, spline: j + 1 }); } else o.push({ name: v.name, v: v.name }); }
  return o;
}
// filas utilizables con X (omite valores perdidos y niveles no vistos en el desarrollo)
export function makeX(ds, meta, info, rows) {
  const X = [], used = [], unseen = {};
  for (const i of rows) {
    const r = []; let ok = true;
    for (const v of info.vars) {
      const val = ds.cols[v.name]?.[i];
      if (val === undefined || isMiss(val)) { ok = false; break; }
      if (v.kind === 'cat') { if (!v.levels.includes(String(val))) { unseen[v.name] = (unseen[v.name] || 0) + 1; ok = false; break; } for (const l of v.levels.slice(1)) r.push(String(val) === l ? 1 : 0); }
      else if (v.knots) r.push(...rcsRow(val, v.knots)); else r.push(val);
    }
    if (ok) { X.push(Float64Array.from(r)); used.push(i); }
  }
  return { X, used, unseen };
}
export const eventVec = (ds, spec, rows) => Float64Array.from(rows, i => (String(ds.cols[spec.y][i]) === String(spec.event) ? 1 : 0));
const withIntercept = X => X.map(x => Float64Array.from([1, ...x]));

// filas con datos completos en todas las variables del modelo
export function completeFor(ds, spec) {
  const vars = [...spec.vars.map(v => v.name), ...(spec.type === 'survival' ? [spec.time, spec.status] : [spec.y])], out = [];
  for (let i = 0; i < ds.n; i++) if (vars.every(v => !isMiss(ds.cols[v][i]))) out.push(i);
  return out;
}
const statusVec = (ds, spec, rows) => Float64Array.from(rows, i => (String(ds.cols[spec.status][i]) === String(spec.event) ? 1 : 0));
const timeVec = (ds, spec, rows) => Float64Array.from(rows, i => +ds.cols[spec.time][i]);

// ---------------------------------------------------------------- ajuste
// λ de ridge para Cox por validación cruzada de la log-verosimilitud parcial (van Houwelingen)
function cvCoxRidge(Z, time, status, { folds = 10, seed = 1 } = {}) {
  const n = Z.length, R = rng(seed), grid = Array.from({ length: 16 }, (_, k) => 0.05 * Math.pow(1.6, k) * Math.sqrt(n) / 10), idx = Array.from({ length: n }, (_, i) => i).sort(() => R.u() - 0.5), fold = new Int32Array(n);
  idx.forEach((i, k) => (fold[i] = k % folds));
  const score = grid.map(lam => { let s = 0; for (let k = 0; k < folds; k++) { const tr = idx.filter(i => fold[i] !== k); const Xt = tr.map(i => Z[i]), tt = Float64Array.from(tr, i => time[i]), st = Float64Array.from(tr, i => status[i]); try { const f = fitCox(Xt, tt, st, { ridge: lam }); s += coxLogLik(Z, time, status, f.beta) - coxLogLik(Xt, tt, st, f.beta); } catch { s -= 1e9; } } return s; });
  let ib = 0; score.forEach((v, i) => { if (v > score[ib]) ib = i; });
  return { lambda: grid[ib], grid, score };
}
export function breslow(lp, time, status) { // riesgo acumulado basal H0(t) con LP centrado
  const idx = Array.from(time.keys()).sort((a, b) => time[b] - time[a]), t = [], H = []; let S = 0; const steps = [];
  for (let k = 0; k < idx.length;) { let j = k, d = 0, add = 0; while (j < idx.length && time[idx[j]] === time[idx[k]]) { S += Math.exp(lp[idx[j]]); if (status[idx[j]]) d++; j++; } if (d) steps.push([time[idx[k]], d / S]); k = j; }
  steps.reverse(); let c = 0; for (const [tt, h] of steps) { c += h; t.push(tt); H.push(c); }
  return { t, H };
}
export const H0at = (base, t) => { let lo = 0, hi = base.t.length - 1, ans = 0; while (lo <= hi) { const m = (lo + hi) >> 1; if (base.t[m] <= t) { ans = base.H[m]; lo = m + 1; } else hi = m - 1; } return ans; };

export function fitPred(ds, meta, spec, rows, { seed = 1, folds = 10, quiet = false } = {}) {
  const info = designInfo(ds, meta, spec, rows), cols = columnNames(info), { X, used } = makeX(ds, meta, info, rows);
  if (used.length < cols.length + 5) throw new Error('Muy pocas filas completas para el número de parámetros del modelo.');
  const m = { type: spec.type, info, cols, spec: { ...spec, vars: spec.vars.map(v => ({ ...v })) }, n: used.length, penalty: spec.penalty || 'none', rows: used };
  let pen = null;
  if (spec.type === 'binary' || spec.type === 'continuous') {
    const y = spec.type === 'binary' ? eventVec(ds, spec, used) : Float64Array.from(used, i => +ds.cols[spec.y][i]), fam = spec.type === 'binary' ? 'binomial' : 'gaussian';
    m.events = spec.type === 'binary' ? sum(y) : null;
    if (spec.penalty && spec.penalty !== 'none') {
      const alpha = spec.penalty === 'lasso' ? 1 : 0, cv = cvEnet(X, y, { family: fam, alpha, nfolds: folds, nlambda: 40, seed }), lam = spec.lambdaRule === '1se' ? cv.lambda1se : cv.lambdaMin, f = enet(X, y, { family: fam, alpha, lambda: lam });
      m.intercept = f.intercept; m.beta = f.beta; pen = { lambda: lam, rule: spec.lambdaRule || 'min', df: f.df, cv: quiet ? null : { lambdas: cv.lambdas, cv: cv.cv, se: cv.se, imin: cv.imin, i1: cv.i1 } };
    } else if (fam === 'binomial') { const f = fitGLM(withIntercept(X), y, 'binomial'); m.intercept = f.beta[0]; m.beta = Float64Array.from(f.beta.slice(1)); m.se = [f.se[0], ...f.se.slice(1)]; m.sep = f.sep; m.fit = f; }
    else { const f = fitOLS(withIntercept(X), y); m.intercept = f.beta[0]; m.beta = Float64Array.from(f.beta.slice(1)); m.se = Array.from(f.se); m.sigma = f.sigma; m.fit = f; }
    if (fam === 'gaussian') m.sigma ??= Math.sqrt(sum(Array.from(y, (v, k) => (v - m.intercept - X[k].reduce((s, a, j) => s + a * m.beta[j], 0)) ** 2)) / (y.length - cols.length - 1));
  } else {
    const t = timeVec(ds, spec, used), st = statusVec(ds, spec, used); m.events = sum(st);
    if (spec.penalty === 'ridge') { const S = standardize(X), cv = cvCoxRidge(S.Z, t, st, { folds, seed }), f = fitCox(S.Z, t, st, { ridge: cv.lambda }); m.beta = Float64Array.from(f.beta, (b, j) => b / S.sd[j]); pen = { lambda: cv.lambda, rule: 'CVPL', df: cols.length, cv: quiet ? null : { grid: cv.grid, score: cv.score } }; }
    else { const f = fitCox(X, t, st); m.beta = f.beta; m.se = Array.from(f.se); m.sep = f.sep; m.fit = f; }
    const lp = X.map(x => x.reduce((s, a, j) => s + a * m.beta[j], 0)); m.lpMean = sum(lp) / lp.length; const lc = Float64Array.from(lp, v => v - m.lpMean); m.base = breslow(lc, t, st);
    m.tMax = Math.max(...t);
  }
  m.pen = pen; return m;
}
// predictor lineal en los datos `ds` (filas `rows`) con el modelo ajustado; devuelve LP y las filas usadas
export function predictLPs(m, ds, meta, rows) {
  const { X, used, unseen } = makeX(ds, meta, m.info, rows), lp = Float64Array.from(X, x => { let s = m.type === 'survival' ? 0 : m.intercept; for (let j = 0; j < x.length; j++) s += x[j] * m.beta[j]; return s; });
  return { lp, used, unseen };
}
export const riskFromLP = (m, lp, horizon) => (m.type === 'binary' ? Float64Array.from(lp, expit) : m.type === 'survival' ? Float64Array.from(lp, v => 1 - Math.exp(-H0at(m.base, horizon) * Math.exp(v - m.lpMean))) : lp);

// ---------------------------------------------------------------- índice C de Harrell en O(n log n)
export function cindex(lp, time, status) {
  const n = lp.length, ord = Array.from({ length: n }, (_, i) => i).sort((a, b) => time[b] - time[a] || status[a] - status[b]);
  const vals = Array.from(new Set(Array.from(lp))).sort((a, b) => a - b), rank = new Map(vals.map((v, i) => [v, i + 1])), m = vals.length, bit = new Float64Array(m + 2);
  const add = r => { for (; r <= m; r += r & -r) bit[r]++; }, q = r => { let s = 0; for (; r > 0; r -= r & -r) s += bit[r]; return s; };
  let conc = 0, ties = 0, comp = 0, total = 0;
  for (let k = 0; k < n;) {
    let j = k; while (j + 1 < n && time[ord[j + 1]] === time[ord[k]]) j++;
    const grp = ord.slice(k, j + 1), ev = grp.filter(i => status[i]), ce = grp.filter(i => !status[i]);
    for (const i of ev) { // parejas con sujetos de tiempo mayor (ya insertados) y con censurados del mismo tiempo
      const r = rank.get(lp[i]), below = q(r - 1), upto = q(r), tie = upto - below, above = total - upto; comp += total; conc += below; ties += tie; // below: lp menor ⇒ concordante (el evento temprano tiene mayor riesgo)
      for (const c of ce) { comp++; if (lp[c] < lp[i]) conc++; else if (lp[c] === lp[i]) ties++; }
    }
    for (const i of grp) { add(rank.get(lp[i])); total++; }
    k = j + 1;
  }
  return { c: (conc + 0.5 * ties) / comp, pairs: comp };
}

// ---------------------------------------------------------------- medidas de rendimiento
export function aucOf(score, y) {
  const idx = Array.from(score.keys()).sort((a, b) => score[a] - score[b]); let r = 0, n1 = 0;
  for (let k = 0; k < idx.length;) { let j = k; while (j + 1 < idx.length && score[idx[j + 1]] === score[idx[k]]) j++; const rr = (k + j) / 2 + 1; for (let m = k; m <= j; m++) if (y[idx[m]]) { r += rr; n1++; } k = j + 1; }
  const n0 = idx.length - n1; return (r - n1 * (n1 + 1) / 2) / (n1 * n0);
}
export function aucCI(A, n1, n0) { // Hanley-McNeil
  const q1 = A / (2 - A), q2 = 2 * A * A / (1 + A), se = Math.sqrt((A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n0 - 1) * (q2 - A * A)) / (n1 * n0)), z = qnorm(0.975);
  return [Math.max(0, A - z * se), Math.min(1, A + z * se), se];
}
export function perfBinary(p, y) {
  const n = y.length, e = sum(y), prev = e / n, lp = Float64Array.from(p, v => logit(clampP(v)));
  const out = { n, events: e, prev, auc: aucOf(p, y), brier: sum(Array.from(p, (v, i) => (v - y[i]) ** 2)) / n };
  [out.aucLo, out.aucHi] = aucCI(out.auc, e, n - e); out.brierScaled = 1 - out.brier / (prev * (1 - prev)); out.oe = e / sum(p);
  try { const f = fitGLM(Array.from(lp, v => Float64Array.of(1, v)), y, 'binomial'); out.slope = f.beta[1]; out.slopeSE = f.se[1]; out.interceptRecal = f.beta[0]; } catch { out.slope = NaN; out.slopeSE = NaN; }
  try { const f = fitGLM(Array.from(lp, () => Float64Array.of(1)), y, 'binomial', { offset: lp }); out.citl = f.beta[0]; out.citlSE = f.se[0]; } catch { out.citl = NaN; out.citlSE = NaN; }
  const ll1 = sum(Array.from(p, (v, i) => (y[i] ? Math.log(clampP(v)) : Math.log(1 - clampP(v))))), ll0 = n * (prev * Math.log(prev) + (1 - prev) * Math.log(1 - prev));
  out.r2cs = 1 - Math.exp(2 * (ll0 - ll1) / n); out.r2nag = out.r2cs / (1 - Math.exp(2 * ll0 / n)); out.ll = ll1;
  return out;
}
export function perfSurvival(lp, time, status, { base = null, lpMean = 0, horizon = null } = {}) {
  const n = lp.length, c = cindex(lp, time, status), out = { n, events: sum(status), cindex: c.c };
  try { const f = fitCox(Array.from(lp, v => Float64Array.of(v)), time, status); out.slope = f.beta[0]; out.slopeSE = f.se[0]; out.r2cs = 1 - Math.exp(-f.lr / n); out.lr = f.lr; } catch { out.slope = NaN; out.slopeSE = NaN; }
  if (base && horizon) {
    const H = H0at(base, horizon), risk = Float64Array.from(lp, v => 1 - Math.exp(-H * Math.exp(v - lpMean))), k = km(time, status), at = k.steps.filter(s => s.t <= horizon).pop(), obs = at ? 1 - at.surv : 0;
    out.horizon = horizon; out.expected = sum(risk) / n; out.observed = obs; out.oe = obs / out.expected; out.risk = risk;
    // Brier ponderado por la probabilidad inversa de censura (estimador de Graf et al.)
    const kc = km(time, Float64Array.from(status, s => 1 - s)), G = t => { const s = kc.steps.filter(x => x.t <= t).pop(); return s ? s.surv : 1; };
    let b = 0; for (let i = 0; i < n; i++) { if (time[i] <= horizon && status[i]) b += (1 - risk[i]) ** 2 / Math.max(G(time[i]), 1e-3); else if (time[i] > horizon) b += risk[i] ** 2 / Math.max(G(horizon), 1e-3); }
    out.brier = b / n;
  }
  return out;
}
export function perfContinuous(pred, y) {
  const n = y.length, my = sum(y) / n, sst = sum(Array.from(y, v => (v - my) ** 2)), sse = sum(Array.from(y, (v, i) => (v - pred[i]) ** 2)), out = { n, r2: 1 - sse / sst, rmse: Math.sqrt(sse / n), mae: sum(Array.from(y, (v, i) => Math.abs(v - pred[i]))) / n, citl: my - sum(pred) / n };
  try { const f = fitOLS(Array.from(pred, v => Float64Array.of(1, v)), y); out.slope = f.beta[1]; out.slopeSE = f.se[1]; out.intercept = f.beta[0]; } catch { out.slope = NaN; }
  return out;
}
// rendimiento de un modelo en un conjunto de datos
export function evaluate(m, ds, meta, rows, { horizon = null } = {}) {
  const { lp, used, unseen } = predictLPs(m, ds, meta, rows);
  if (m.type === 'binary') { const y = eventVec(ds, m.spec, used), p = Float64Array.from(lp, expit); return { perf: perfBinary(p, y), lp, p, y, used, unseen }; }
  if (m.type === 'continuous') { const y = Float64Array.from(used, i => +ds.cols[m.spec.y][i]); return { perf: perfContinuous(lp, y), lp, p: lp, y, used, unseen }; }
  const t = timeVec(ds, m.spec, used), s = statusVec(ds, m.spec, used), h = horizon || m.spec.horizon; const perf = perfSurvival(lp, t, s, { base: m.base, lpMean: m.lpMean, horizon: h });
  return { perf, lp, p: perf.risk, time: t, status: s, used, unseen };
}

// ---------------------------------------------------------------- validación interna por bootstrap (optimismo de Harrell)
const KEYS = { binary: ['auc', 'slope', 'citl', 'brier', 'r2cs'], survival: ['cindex', 'slope', 'r2cs'], continuous: ['r2', 'slope', 'rmse'] };
const pick = (perf, ks) => Object.fromEntries(ks.map(k => [k, perf[k]]));
export async function bootstrapValidate(ds, meta, spec, rows, model, { B = 200, seed = 99, horizon = null, onProgress = null, retune = true, calib = true } = {}) {
  const R = rng(seed), ev0 = evaluate(model, ds, meta, rows, { horizon }), ks = KEYS[spec.type], app = pick(ev0.perf, ks), idxAll = model.rows, n = idxAll.length;
  const opt = Object.fromEntries(ks.map(k => [k, []])), slopes = []; let failed = 0;
  // curva de calibración corregida por optimismo (binario): suavizado lowess en una rejilla común de riesgos
  const doCal = calib && spec.type === 'binary', NG = 40, sortedP = doCal ? Array.from(ev0.p).sort((a, b) => a - b) : null, glo = doCal ? quantile(sortedP, 0.02) : 0, ghi = doCal ? quantile(sortedP, 0.98) : 1;
  const calDiff = doCal ? new Float64Array(NG) : null; let calN = 0;
  for (let b = 0; b < B; b++) {
    const samp = Array.from({ length: n }, () => idxAll[R.int(n)]);
    try {
      const mb = !retune && model.pen && spec.penalty !== 'none' ? fitFixedLambda(ds, meta, spec, samp, model.pen.lambda) : fitPred(ds, meta, spec, samp, { seed: seed + b, quiet: true });
      const ea = evaluate(mb, ds, meta, samp, { horizon }), et = evaluate(mb, ds, meta, idxAll, { horizon }), ab = pick(ea.perf, ks), te = pick(et.perf, ks);
      if (ks.some(k => !isFinite(ab[k]) || !isFinite(te[k]))) throw new Error('métrica no finita');
      if (doCal) { const ca = lowess(ea.p, ea.y, 2 / 3, NG, glo, ghi), ct = lowess(et.p, et.y, 2 / 3, NG, glo, ghi); if (ca.length === NG && ct.length === NG) { for (let g = 0; g < NG; g++) calDiff[g] += ca[g][1] - ct[g][1]; calN++; } }
      for (const k of ks) opt[k].push(ab[k] - te[k]); slopes.push(te.slope);
    } catch { failed++; }
    if (b % 5 === 4) { onProgress?.((b + 1) / B); await tick(); }
  }
  const mean = a => sum(a) / a.length, res = ks.map(k => { const o = mean(opt[k]); return { key: k, apparent: app[k], optimism: o, corrected: app[k] - o, sdOpt: Math.sqrt(sum(opt[k].map(v => (v - o) ** 2)) / Math.max(opt[k].length - 1, 1)) }; });
  let calibration = null;
  if (doCal && calN > 5) { const c0 = lowess(ev0.p, ev0.y, 2 / 3, NG, glo, ghi); calibration = { apparent: c0, corrected: c0.map(([x, y], g) => [x, Math.min(1, Math.max(0, y - calDiff[g] / calN))]) }; }
  return { rows: res, B: B - failed, failed, shrinkage: slopes.length ? mean(slopes) : NaN, slopes, calibration };
}
function fitFixedLambda(ds, meta, spec, rows, lambda) {
  const info = designInfo(ds, meta, spec, rows), cols = columnNames(info), { X, used } = makeX(ds, meta, info, rows), m = { type: spec.type, info, cols, spec, n: used.length, rows: used, penalty: spec.penalty };
  if (spec.type === 'survival') { const t = timeVec(ds, spec, used), st = statusVec(ds, spec, used), S = standardize(X), f = fitCox(S.Z, t, st, { ridge: lambda }); m.beta = Float64Array.from(f.beta, (b, j) => b / S.sd[j]); const lp = X.map(x => x.reduce((s, a, j) => s + a * m.beta[j], 0)); m.lpMean = sum(lp) / lp.length; m.base = breslow(Float64Array.from(lp, v => v - m.lpMean), t, st); return m; }
  const fam = spec.type === 'binary' ? 'binomial' : 'gaussian', y = spec.type === 'binary' ? eventVec(ds, spec, used) : Float64Array.from(used, i => +ds.cols[spec.y][i]), f = enet(X, y, { family: fam, alpha: spec.penalty === 'lasso' ? 1 : 0, lambda });
  m.intercept = f.intercept; m.beta = f.beta; return m;
}

// ---------------------------------------------------------------- contracción uniforme y recalibración
// modelo con coeficientes multiplicados por γ; el intercepto (binario/lineal) o el riesgo basal (Cox) se vuelve a estimar
export function shrinkModel(m, gamma, ds, meta) {
  const s = { ...m, beta: Float64Array.from(m.beta, b => b * gamma), shrunk: gamma, fit: null, se: null };
  const ev = evaluate(m, ds, meta, m.rows), lp0 = Float64Array.from(ev.lp, v => v - (m.type === 'survival' ? 0 : m.intercept));
  if (m.type === 'binary') { const f = fitGLM(Array.from(lp0, () => Float64Array.of(1)), ev.y, 'binomial', { offset: Float64Array.from(lp0, v => gamma * v) }); s.intercept = f.beta[0]; }
  else if (m.type === 'continuous') { const my = sum(ev.y) / ev.y.length, ms = sum(lp0) / lp0.length; s.intercept = my - gamma * ms; }
  else { const lp = Float64Array.from(lp0, v => gamma * v); s.lpMean = sum(lp) / lp.length; s.base = breslow(Float64Array.from(lp, v => v - s.lpMean), ev.time, ev.status); }
  return s;
}
// recalibración en datos nuevos: 'citl' (solo intercepto), 'slope' (intercepto y pendiente)
export function recalibrate(m, ds, meta, rows, kind = 'slope') {
  const ev = evaluate(m, ds, meta, rows), out = { ...m, recal: kind, fit: null, se: null };
  if (m.type === 'binary') {
    if (kind === 'citl') { const f = fitGLM(Array.from(ev.lp, () => Float64Array.of(1)), ev.y, 'binomial', { offset: ev.lp }); out.intercept = m.intercept + f.beta[0]; }
    else { const f = fitGLM(Array.from(ev.lp, v => Float64Array.of(1, v)), ev.y, 'binomial'); out.intercept = f.beta[0] + f.beta[1] * m.intercept; out.beta = Float64Array.from(m.beta, b => b * f.beta[1]); out.recalSlope = f.beta[1]; }
  } else if (m.type === 'continuous') { const f = fitOLS(Array.from(ev.lp, v => Float64Array.of(1, v)), ev.y); if (kind === 'citl') out.intercept = m.intercept + (sum(ev.y) / ev.y.length - sum(ev.lp) / ev.lp.length); else { out.intercept = f.beta[0] + f.beta[1] * m.intercept; out.beta = Float64Array.from(m.beta, b => b * f.beta[1]); out.recalSlope = f.beta[1]; } }
  else { // Cox: nuevo riesgo basal en los datos nuevos; con 'slope' además se reescalan los coeficientes
    let g = 1; if (kind === 'slope') { const f = fitCox(Array.from(ev.lp, v => Float64Array.of(v)), ev.time, ev.status); g = f.beta[0]; out.beta = Float64Array.from(m.beta, b => b * g); out.recalSlope = g; }
    const lp = Float64Array.from(ev.lp, v => g * v); out.lpMean = sum(lp) / lp.length; out.base = breslow(Float64Array.from(lp, v => v - out.lpMean), ev.time, ev.status);
  }
  return out;
}

// ---------------------------------------------------------------- calibración
export function wilson(k, n, z = 1.96) { const p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n), h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [(c - h) / d, (c + h) / d]; }
export function lowess(x, y, f = 2 / 3, npts = 60, lo = null, hi = null) {
  const idx = Array.from(x.keys()).filter(i => isFinite(x[i]) && isFinite(y[i])).sort((a, b) => x[a] - x[b]), xs = idx.map(i => x[i]), ys = idx.map(i => y[i]), n = xs.length; if (n < 5) return [];
  const r = Math.max(3, Math.ceil(f * n)), out = [], a0 = lo ?? xs[0], b0 = hi ?? xs[n - 1];
  for (let k = 0; k < npts; k++) {
    const x0 = a0 + (b0 - a0) * k / (npts - 1), d = xs.map(v => Math.abs(v - x0)), h = [...d].sort((a, b) => a - b)[r - 1] || 1e-12; let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { const u = d[i] / h; if (u >= 1) continue; const w = (1 - u ** 3) ** 3; sw += w; sx += w * xs[i]; sy += w * ys[i]; sxx += w * xs[i] ** 2; sxy += w * xs[i] * ys[i]; }
    const den = sw * sxx - sx * sx, bb = Math.abs(den) > 1e-12 ? (sw * sxy - sx * sy) / den : 0, a = (sy - bb * sx) / sw; out.push([x0, a + bb * x0]);
  }
  return out;
}
export function calibGroups(p, y, g = 10) { // grupos de igual tamaño por riesgo predicho
  const idx = Array.from(p.keys()).sort((a, b) => p[a] - p[b]), n = idx.length, rows = [];
  for (let k = 0; k < g; k++) { const s = idx.slice(Math.floor(k * n / g), Math.floor((k + 1) * n / g)); if (!s.length) continue; const ev = sum(s.map(i => y[i])), [lo, hi] = wilson(ev, s.length); rows.push({ pred: sum(s.map(i => p[i])) / s.length, obs: ev / s.length, lo, hi, n: s.length }); }
  return rows;
}
export function calibSurvival(risk, time, status, horizon, g = 5) {
  const idx = Array.from(risk.keys()).sort((a, b) => risk[a] - risk[b]), n = idx.length, rows = [];
  for (let k = 0; k < g; k++) { const s = idx.slice(Math.floor(k * n / g), Math.floor((k + 1) * n / g)); if (s.length < 5) continue; const kmf = km(Float64Array.from(s, i => time[i]), Float64Array.from(s, i => status[i])), at = kmf.steps.filter(x => x.t <= horizon).pop(); if (!at) continue; rows.push({ pred: sum(s.map(i => risk[i])) / s.length, obs: 1 - at.surv, lo: 1 - at.hi, hi: 1 - at.lo, n: s.length }); }
  return rows;
}

// ---------------------------------------------------------------- análisis de curva de decisión
export function dcaBinary(p, y, thresholds) {
  const n = y.length, prev = sum(y) / n;
  return thresholds.map(pt => { let tp = 0, fp = 0; for (let i = 0; i < n; i++) if (p[i] >= pt) { if (y[i]) tp++; else fp++; } const w = pt / (1 - pt), nb = tp / n - fp / n * w; return { pt, model: nb, all: prev - (1 - prev) * w, none: 0, avoid: (nb - (prev - (1 - prev) * w)) / w * 100, tp, fp, treated: (tp + fp) / n }; });
}
export function dcaSurvival(risk, time, status, horizon, thresholds) {
  const n = risk.length, kAll = km(time, status), sAll = (kAll.steps.filter(s => s.t <= horizon).pop() || { surv: 1 }).surv, prev = 1 - sAll;
  return thresholds.map(pt => {
    const sel = []; for (let i = 0; i < n; i++) if (risk[i] >= pt) sel.push(i); const pos = sel.length / n; let nb = 0;
    if (sel.length >= 5) { const k = km(Float64Array.from(sel, i => time[i]), Float64Array.from(sel, i => status[i])), s = (k.steps.filter(x => x.t <= horizon).pop() || { surv: 1 }).surv, tp = (1 - s) * pos, fp = s * pos; nb = tp - fp * pt / (1 - pt); }
    const w = pt / (1 - pt), all = prev - (1 - prev) * w; return { pt, model: nb, all, none: 0, avoid: (nb - all) / w * 100, treated: pos };
  });
}

// ---------------------------------------------------------------- modelo publicado (ecuación): LP = a + Σ β·x
export function evalEquation(eq, ds, rows) {
  const lp = [], used = [];
  for (const i of rows) { let s = eq.type === 'survival' ? 0 : +eq.intercept || 0, ok = true; for (const t of eq.terms) { const v = ds.cols[t.name]?.[i]; if (v === undefined) throw new Error(`La base no tiene la variable «${t.name}».`); const x = typeof v === 'number' ? v : parseFloat(v); if (!isFinite(x)) { ok = false; break; } s += t.coef * x; } if (ok) { lp.push(s); used.push(i); } }
  return { lp: Float64Array.from(lp), used };
}
