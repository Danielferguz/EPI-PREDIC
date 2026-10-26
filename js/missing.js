// Datos perdidos: patrones, prueba MCAR de Little, exploración MAR, imputación múltiple (MICE) y reglas de Rubin.
import { isMiss, design, levelsOf } from './data.js?v=2026.10.26-1';
import { crossW, crossWv, symInv, mv, dot, chol, zeros, sub, quadForm, logdet } from './linalg.js?v=2026.10.26-1';
import { fitGLM } from './models.js?v=2026.10.26-1';
import { rng, pchisqU, pfU, qt, p2t } from './dist.js?v=2026.10.26-1';
import { welch, chisqTest } from './tests.js?v=2026.10.26-1';

// ---------------- resumen y patrones ----------------
export function missingSummary(ds, vars) {
  return vars.map(v => { let k = 0; for (const x of ds.cols[v]) if (isMiss(x)) k++; return { v, n: ds.n, miss: k, pct: k / ds.n }; });
}
// patrones (como md.pattern de mice): filas = patrones, 1 = observado
export function patterns(ds, vars) {
  const map = new Map();
  for (let i = 0; i < ds.n; i++) { const key = vars.map(v => (isMiss(ds.cols[v][i]) ? 0 : 1)).join(''); map.set(key, (map.get(key) || 0) + 1); }
  return [...map.entries()].map(([k, n]) => ({ obs: k.split('').map(Number), n, nmiss: k.split('').filter(c => c === '0').length })).sort((a, b) => a.nmiss - b.nmiss || b.n - a.n);
}

// ---------------- prueba MCAR de Little (1988) ----------------
// vars cuantitativas (las categóricas binarias se codifican 0/1); EM para normal multivariante
export function littleMCAR(ds, meta, vars) {
  const p = vars.length, rows = [];
  const enc = vars.map(v => { const m = meta[v]; if (m.type === 'num') return x => (isMiss(x) ? NaN : x); const L = levelsOf(m); return x => (isMiss(x) ? NaN : L.indexOf(x)); });
  for (let i = 0; i < ds.n; i++) { const r = vars.map((v, j) => enc[j](ds.cols[v][i])); if (r.some(isFinite)) rows.push(r); }
  const n = rows.length;
  // agrupar por patrón
  const pats = new Map(); rows.forEach(r => { const k = r.map(x => (isFinite(x) ? 1 : 0)).join(''); if (!pats.has(k)) pats.set(k, []); pats.get(k).push(r); });
  const P = [...pats.entries()].map(([k, rs]) => ({ o: [...k].map((c, j) => (c === '1' ? j : -1)).filter(j => j >= 0), rs }));
  let mu = vars.map((_, j) => { const o = rows.map(r => r[j]).filter(isFinite); return o.reduce((a, b) => a + b, 0) / o.length; });
  let S = zeros(p, p); vars.forEach((_, j) => { const o = rows.map(r => r[j]).filter(isFinite); S[j][j] = o.reduce((a, b) => a + (b - mu[j]) ** 2, 0) / o.length || 1; });
  let it, conv = false;
  for (it = 0; it < 1000; it++) {
    const T1 = new Float64Array(p), T2 = zeros(p, p);
    for (const { o, rs } of P) {
      const m = vars.map((_, j) => j).filter(j => !o.includes(j)), Soi = symInv(sub(S, o));
      const B = m.map(a => mv(Soi, o.map(b => S[a][b]))); // Σ_mo Σ_oo^-1
      const C = m.map((a, ia) => m.map(b => S[a][b] - dot(B[ia], o.map(c => S[c][b]))));
      for (const r of rs) {
        const x = Float64Array.from(r), d = o.map(j => r[j] - mu[j]);
        m.forEach((a, ia) => { x[a] = mu[a] + dot(B[ia], d); });
        for (let a = 0; a < p; a++) { T1[a] += x[a]; for (let b = 0; b < p; b++) T2[a][b] += x[a] * x[b]; }
        m.forEach((a, ia) => m.forEach((b, ib) => { T2[a][b] += C[ia][ib]; }));
      }
    }
    const nmu = Array.from(T1, v => v / n), nS = T2.map((r, a) => r.map((v, b) => v / n - nmu[a] * nmu[b]));
    const delta = Math.max(...nmu.map((v, j) => Math.abs(v - mu[j])), ...nS.flatMap((r, a) => Array.from(r, (v, b) => Math.abs(v - S[a][b]))));
    const scl = Math.max(1, ...nS.map((r, a) => Math.abs(r[a])));
    mu = nmu; S = nS; if (delta < 1e-9 * scl) { conv = true; break; }
  }
  let d2 = 0, df = 0;
  for (const { o, rs } of P) {
    const ybar = o.map(j => rs.reduce((a, r) => a + r[j], 0) / rs.length), dv = ybar.map((v, k) => v - mu[o[k]]);
    const Si = symInv(sub(S, o)); if (!Si) continue;
    d2 += rs.length * quadForm(dv, Si); df += o.length;
  }
  df -= p;
  return { stat: d2, df, p: df > 0 ? pchisqU(d2, df) : NaN, npat: P.length, n, mu, S, conv, iter: it };
}

// ---------------- exploración del mecanismo (MAR) ----------------
// compara cada variable entre filas con y sin dato perdido en la variable objetivo
export function compareByMissing(ds, meta, target, others) {
  const R = Array.from(ds.cols[target], x => isMiss(x));
  return others.filter(w => w !== target).map(w => {
    const m = meta[w], col = ds.cols[w];
    if (m.type === 'num') {
      const a = [], b = []; for (let i = 0; i < ds.n; i++) if (!isMiss(col[i])) (R[i] ? a : b).push(col[i]);
      if (a.length < 2 || b.length < 2) return { w, test: '–', p: NaN };
      const t = welch(a, b); return { w, test: 'Welch t', p: t.p, sMiss: mean(a), sObs: mean(b), kind: 'num' };
    }
    const L = levelsOf(m), tab = [0, 1].map(r => L.map(l => { let k = 0; for (let i = 0; i < ds.n; i++) if (col[i] === l && R[i] === !!r) k++; return k; }));
    if (tab[1].reduce((a, b) => a + b, 0) === 0) return { w, test: '–', p: NaN };
    const c = chisqTest(tab); return { w, test: 'χ²', p: c.p, sMiss: tab[1], sObs: tab[0], kind: 'cat', levels: L };
  });
}
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
// regresión logística del indicador de perdido sobre las demás variables (filas completas en los predictores)
export function missingnessModel(ds, meta, target, preds) {
  const pv = preds.filter(w => w !== target), rows = [];
  for (let i = 0; i < ds.n; i++) if (pv.every(w => !isMiss(ds.cols[w][i]))) rows.push(i);
  const y = Float64Array.from(rows, i => (isMiss(ds.cols[target][i]) ? 1 : 0)), ev = y.reduce((a, b) => a + b, 0);
  if (ev < 3 || ev > rows.length - 3) return null;
  const D = design(ds, meta, pv.map(v => ({ vars: [v] })), rows);
  try {
    const f = fitGLM(D.X, y, 'binomial'), lr = 2 * (f.ll - f.ll0), df = D.X[0].length - 1;
    return { fit: f, D, n: rows.length, events: ev, lr, df, p: pchisqU(lr, df), auc: auc(f.fitted, y) };
  } catch (e) { return { error: e.message }; }
}
export function auc(score, y) {
  const idx = Array.from(score.keys()).sort((a, b) => score[a] - score[b]); let r = 0, n1 = 0;
  for (let k = 0; k < idx.length;) { let j = k; while (j + 1 < idx.length && score[idx[j + 1]] === score[idx[k]]) j++; const rr = (k + j) / 2 + 1; for (let m = k; m <= j; m++) if (y[idx[m]]) { r += rr; n1++; } k = j + 1; }
  const n0 = idx.length - n1; return (r - n1 * (n1 + 1) / 2) / (n1 * n0);
}

// ---------------- imputación simple ----------------
// imputación simple: mean (media/moda), median (mediana/moda), knn (k vecinos), locf (última observación, datos longitudinales), missind (categoría «Perdido»)
export function singleImpute(ds, meta, vars, method = 'mean', { id = null, tvar = null, k = 5 } = {}) {
  if (method === 'mean') return simpleImpute(ds, meta, vars);
  const cols = { ...ds.cols };
  if (method === 'median' || method === 'missind') {
    for (const v of vars) {
      const c = ds.cols[v], m = meta[v];
      if (m.type === 'num') { const o = Array.from(c).filter(isFinite).sort((a, b) => a - b), md = o.length % 2 ? o[(o.length - 1) / 2] : (o[o.length / 2 - 1] + o[o.length / 2]) / 2; cols[v] = Float64Array.from(c, x => (isFinite(x) ? x : md)); }
      else if (method === 'missind') cols[v] = c.map(x => x ?? 'Perdido');
      else { const cnt = {}; c.forEach(x => x != null && (cnt[x] = (cnt[x] || 0) + 1)); const mode = Object.entries(cnt).sort((a, b) => b[1] - a[1])[0]?.[0]; cols[v] = c.map(x => x ?? mode); }
    }
    return { n: ds.n, cols };
  }
  if (method === 'locf') {
    if (!id) throw new Error('LOCF requiere un identificador de sujeto (datos longitudinales).');
    const order = Array.from({ length: ds.n }, (_, i) => i).sort((a, b) => String(ds.cols[id][a]).localeCompare(String(ds.cols[id][b]), 'es', { numeric: true }) || (tvar ? ds.cols[tvar][a] - ds.cols[tvar][b] : a - b));
    for (const v of vars) {
      const c = ds.cols[v], o = c instanceof Float64Array ? Float64Array.from(c) : c.slice();
      let prevId = null, last = null;
      for (const i of order) { if (ds.cols[id][i] !== prevId) { prevId = ds.cols[id][i]; last = null; } if (isMiss(o[i])) { if (last != null) o[i] = last; } else last = o[i]; }
      cols[v] = o;
    }
    return { n: ds.n, cols };
  }
  if (method === 'knn') {
    // distancia de Gower entre filas usando las variables observadas en ambas
    const num = vars.filter(v => meta[v].type === 'num'), rng_ = Object.fromEntries(num.map(v => { const o = Array.from(ds.cols[v]).filter(isFinite); return [v, (Math.max(...o) - Math.min(...o)) || 1]; }));
    if (ds.n > 6000) throw new Error('kNN está limitado a 6000 filas; use MICE.');
    const out = Object.fromEntries(vars.map(v => [v, ds.cols[v] instanceof Float64Array ? Float64Array.from(ds.cols[v]) : ds.cols[v].slice()]));
    for (let i = 0; i < ds.n; i++) {
      const miss_ = vars.filter(v => isMiss(ds.cols[v][i])); if (!miss_.length) continue;
      const dist = [];
      for (let j = 0; j < ds.n; j++) {
        if (j === i) continue; let d = 0, w = 0;
        for (const v of vars) { const a = ds.cols[v][i], b = ds.cols[v][j]; if (isMiss(a) || isMiss(b)) continue; d += meta[v].type === 'num' ? Math.abs(a - b) / rng_[v] : a === b ? 0 : 1; w++; }
        if (w) dist.push([d / w, j]);
      }
      dist.sort((a, b) => a[0] - b[0]);
      for (const v of miss_) {
        const don = []; for (const [, j] of dist) { if (!isMiss(ds.cols[v][j])) don.push(ds.cols[v][j]); if (don.length >= k) break; }
        if (!don.length) continue;
        if (meta[v].type === 'num') out[v][i] = mean(don); else { const cnt = {}; don.forEach(x => (cnt[x] = (cnt[x] || 0) + 1)); out[v][i] = Object.entries(cnt).sort((a, b) => b[1] - a[1])[0][0]; }
      }
    }
    Object.assign(cols, out); return { n: ds.n, cols };
  }
  throw new Error('Método de imputación desconocido: ' + method);
}
export function simpleImpute(ds, meta, vars) {
  const cols = { ...ds.cols };
  for (const v of vars) {
    const c = ds.cols[v], m = meta[v];
    if (m.type === 'num') { const o = Array.from(c).filter(isFinite), mu = mean(o); cols[v] = Float64Array.from(c, x => (isFinite(x) ? x : mu)); }
    else { const cnt = {}; c.forEach(x => x != null && (cnt[x] = (cnt[x] || 0) + 1)); const mode = Object.entries(cnt).sort((a, b) => b[1] - a[1])[0]?.[0]; cols[v] = c.map(x => x ?? mode); }
  }
  return { n: ds.n, cols };
}

// ---------------- MICE ----------------
// opts: {vars, m, maxit, seed, donors, delta:{v, value}, onProgress}
// methods: {var: 'pmm'|'norm'|'mean'|'sample'|'logreg'|'polyreg'} (por defecto pmm / logreg / polyreg)
export async function mice(ds, meta, { vars, m = 20, maxit = 10, seed = 2024, donors = 5, delta = null, predictors = null, methods = {}, onProgress = null } = {}) {
  const R = rng(seed), target = vars.filter(v => ds.cols[v].some(isMiss));
  const missIdx = Object.fromEntries(target.map(v => [v, Array.from(ds.cols[v].keys()).filter(i => isMiss(ds.cols[v][i]))]));
  const obsIdx = Object.fromEntries(target.map(v => [v, Array.from(ds.cols[v].keys()).filter(i => !isMiss(ds.cols[v][i]))]));
  const order = [...target].sort((a, b) => missIdx[a].length - missIdx[b].length);
  const method = Object.fromEntries(target.map(v => [v, methods[v] || defaultMethod(meta[v])]));
  const all = Array.from({ length: ds.n }, (_, i) => i), imps = [], trace = Object.fromEntries(target.map(v => [v, []]));
  for (let k = 0; k < m; k++) {
    const cols = { ...ds.cols };
    for (const v of vars) cols[v] = ds.cols[v] instanceof Float64Array ? Float64Array.from(ds.cols[v]) : ds.cols[v].slice();
    const cur = { n: ds.n, cols };
    for (const v of target) for (const i of missIdx[v]) { const d = obsIdx[v][R.int(obsIdx[v].length)]; cols[v][i] = ds.cols[v][d]; }
    target.forEach(v => trace[v].push([]));
    for (let it = 0; it < maxit; it++) {
      for (const v of order) {
        const pv = (predictors?.[v] || vars).filter(w => w !== v);
        const D = design(cur, meta, pv.map(w => ({ vars: [w] })), all);
        const Xo = obsIdx[v].map(i => D.X[i]), Xm = missIdx[v].map(i => D.X[i]);
        const dv = delta && delta.v === v ? delta.value : 0;
        if (method[v] === 'sample') {
          missIdx[v].forEach(i => { cols[v][i] = ds.cols[v][obsIdx[v][R.int(obsIdx[v].length)]]; if (meta[v].type === 'num') cols[v][i] += dv; });
        } else if (method[v] === 'mean') {
          const yo = obsIdx[v].map(i => ds.cols[v][i]), mu = mean(yo); missIdx[v].forEach(i => { cols[v][i] = mu + dv; });
        } else if (method[v] === 'pmm' || method[v] === 'norm') {
          const yo = obsIdx[v].map(i => ds.cols[v][i]), vals = method[v] === 'pmm' ? pmm(Xo, yo, Xm, R, donors) : normDraw(Xo, yo, Xm, R);
          missIdx[v].forEach((i, j) => { cols[v][i] = vals[j] + dv; });
        } else {
          const L = levelsOf(meta[v]), yo = obsIdx[v].map(i => L.indexOf(ds.cols[v][i]));
          const draw = method[v] === 'logreg' ? logregDraw(Xo, yo, Xm, R, dv) : polyregDraw(Xo, yo, Xm, L.length, R);
          missIdx[v].forEach((i, j) => { cols[v][i] = L[draw[j]]; });
        }
        // resumen para el gráfico de convergencia: media de los imputados (o proporción del nivel no referente)
        const mv_ = missIdx[v].map(i => cols[v][i]);
        trace[v][k].push(meta[v].type === 'num' ? mean(mv_) : mv_.filter(x => x !== meta[v].ref).length / mv_.length);
      }
    }
    imps.push(cur);
    if (onProgress) { onProgress((k + 1) / m); await new Promise(r => setTimeout(r, 0)); }
  }
  return { imps, m, maxit, seed, target, method, trace, missIdx, vars, delta };
}
export const defaultMethod = m => (m.type === 'num' ? 'pmm' : m.levels.length === 2 ? 'logreg' : 'polyreg');
export const METHODS = {
  pmm: 'Emparejamiento por media predictiva (PMM)', norm: 'Regresión lineal bayesiana (norm)', mean: 'Media (no recomendado)', sample: 'Muestra aleatoria de observados',
  logreg: 'Regresión logística', polyreg: 'Logística multinomial',
};
export const methodsFor = m => (m.type === 'num' ? ['pmm', 'norm', 'sample', 'mean'] : m.levels.length === 2 ? ['logreg', 'sample'] : ['polyreg', 'sample']);
// regresión lineal bayesiana: valores = predicción con parámetros extraídos + error aleatorio (mice.impute.norm)
function normDraw(Xo, yo, Xm, R) {
  const p = Xo[0].length, xtx = crossW(Xo); for (let j = 0; j < p; j++) xtx[j][j] *= 1 + 1e-5;
  const V = symInv(xtx), coef = mv(V, crossWv(Xo, null, yo));
  let rss = 0; for (let i = 0; i < Xo.length; i++) rss += (yo[i] - dot(Xo[i], coef)) ** 2;
  const sig = Math.sqrt(rss / R.chisq(Math.max(Xo.length - p, 1))), L = chol(V) || chol(V.map((r, a) => r.map((v, b) => v + (a === b ? 1e-8 : 0))));
  const z = Array.from({ length: p }, () => R.norm()), b = coef.map((c, a) => c + sig * L[a].reduce((s, l, k) => s + l * z[k], 0));
  return Xm.map(x => dot(x, b) + sig * R.norm());
}
// regresión lineal bayesiana (norm.draw de mice) + emparejamiento por media predictiva
function pmm(Xo, yo, Xm, R, donors) {
  const p = Xo[0].length, xtx = crossW(Xo); for (let j = 0; j < p; j++) xtx[j][j] *= 1 + 1e-5;
  const V = symInv(xtx), coef = mv(V, crossWv(Xo, null, yo));
  let rss = 0; for (let i = 0; i < Xo.length; i++) rss += (yo[i] - dot(Xo[i], coef)) ** 2;
  const df = Math.max(Xo.length - p, 1), sig = Math.sqrt(rss / R.chisq(df)), L = chol(V) || chol(V.map((r, a) => r.map((v, b) => v + (a === b ? 1e-8 : 0))));
  const z = Array.from({ length: p }, () => R.norm()), bstar = coef.map((c, a) => c + sig * L[a].reduce((s, l, b) => s + l * z[b], 0));
  const yhatO = Xo.map(x => dot(x, coef)), ord = Array.from(yhatO.keys()).sort((a, b) => yhatO[a] - yhatO[b]), sorted = ord.map(i => yhatO[i]);
  return Xm.map(x => {
    const t = dot(x, bstar);
    // búsqueda binaria y ventana de los k donantes más cercanos
    let lo = 0, hi = sorted.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] < t) lo = mid + 1; else hi = mid; }
    let a = lo - 1, b = lo; const cand = [];
    while (cand.length < Math.min(donors, sorted.length)) { if (b >= sorted.length || (a >= 0 && t - sorted[a] <= sorted[b] - t)) cand.push(ord[a--]); else cand.push(ord[b++]); }
    return yo[cand[R.int(cand.length)]];
  });
}
function logregDraw(Xo, yo, Xm, R, shift = 0) {
  const y = Float64Array.from(yo); let f;
  try { f = fitGLM(Xo, y, 'binomial', { ridge: 1e-4 }); } catch { f = fitGLM(Xo, y, 'binomial', { ridge: 0.1 }); }
  const L = chol(f.Vnaive) || chol(f.Vnaive.map((r, a) => r.map((v, b) => v + (a === b ? 1e-6 : 0))));
  const z = f.beta.map(() => R.norm()), b = Array.from(f.beta, (c, a) => c + (L ? L[a].reduce((s, l, k) => s + l * z[k], 0) : 0));
  return Xm.map(x => (R.u() < 1 / (1 + Math.exp(-(dot(x, b) + shift))) ? 1 : 0));
}
// logística multinomial con remuestreo bootstrap de los observados (incertidumbre de parámetros)
function polyregDraw(Xo, yo, Xm, K, R) {
  const n = Xo.length, bi = Array.from({ length: n }, () => R.int(n));
  const B = fitMultinom(bi.map(i => Xo[i]), bi.map(i => yo[i]), K);
  return Xm.map(x => { const e = [0, ...B.map(b => dot(x, b))], mx = Math.max(...e), w = e.map(v => Math.exp(v - mx)), s = w.reduce((a, c) => a + c, 0); let u = R.u() * s; for (let k = 0; k < K; k++) { u -= w[k]; if (u <= 0) return k; } return K - 1; });
}
export function fitMultinom(X, y, K, ridge = 1e-3) {
  const n = X.length, p = X[0].length, q = (K - 1) * p; let B = Array.from({ length: K - 1 }, () => new Float64Array(p));
  for (let it = 0; it < 50; it++) {
    const g = new Float64Array(q), H = zeros(q, q);
    for (let i = 0; i < n; i++) {
      const x = X[i], e = [0, ...B.map(b => dot(x, b))], mx = Math.max(...e), w = e.map(v => Math.exp(v - mx)), s = w.reduce((a, c) => a + c, 0), pr = w.map(v => v / s);
      for (let a = 1; a < K; a++) {
        const ra = (y[i] === a ? 1 : 0) - pr[a];
        for (let j = 0; j < p; j++) g[(a - 1) * p + j] += ra * x[j];
        for (let b = 1; b < K; b++) { const wab = pr[a] * ((a === b ? 1 : 0) - pr[b]); if (!wab) continue; for (let j = 0; j < p; j++) { const v = wab * x[j]; if (!v) continue; const row = H[(a - 1) * p + j]; for (let l = 0; l < p; l++) row[(b - 1) * p + l] += v * x[l]; } }
      }
    }
    for (let a = 0; a < K - 1; a++) for (let j = 1; j < p; j++) { g[a * p + j] -= ridge * B[a][j]; H[a * p + j][a * p + j] += ridge; }
    const Hi = symInv(H); if (!Hi) break;
    const step = mv(Hi, g); let mx = 0;
    B = B.map((b, a) => b.map((v, j) => { mx = Math.max(mx, Math.abs(step[a * p + j])); return v + step[a * p + j]; }));
    if (mx < 1e-8) break;
  }
  return B;
}

// ---------------- reglas de Rubin (con gl de Barnard-Rubin) ----------------
export function pool(fits, dfcom = Infinity) {
  const m = fits.length, p = fits[0].beta.length;
  const Q = Array.from({ length: p }, (_, j) => mean(fits.map(f => f.beta[j])));
  const U = zeros(p, p), B = zeros(p, p);
  fits.forEach(f => { for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) { U[a][b] += f.V[a][b] / m; B[a][b] += (f.beta[a] - Q[a]) * (f.beta[b] - Q[b]) / (m - 1); } });
  const V = U.map((r, a) => r.map((v, b) => v + (1 + 1 / m) * B[a][b]));
  const se = V.map((r, j) => Math.sqrt(r[j])), dfj = [], fmi = [], lambda = [], riv = [];
  for (let j = 0; j < p; j++) {
    const lam = Math.min(0.9999, Math.max(1e-12, (1 + 1 / m) * B[j][j] / V[j][j])), r = (1 + 1 / m) * B[j][j] / U[j][j];
    const dfold = (m - 1) / lam ** 2, dfobs = isFinite(dfcom) ? (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lam) : Infinity;
    const df = isFinite(dfobs) ? dfold * dfobs / (dfold + dfobs) : dfold;
    dfj.push(df); lambda.push(lam); riv.push(r); fmi.push((r + 2 / (df + 3)) / (r + 1));
  }
  const stat = Q.map((q, j) => q / se[j]);
  return { pooled: true, m, beta: Q, V, U, B, se, dfj, df: NaN, fmi, lambda, riv, stat, p: stat.map((z, j) => p2t(z, dfj[j])) };
}
// prueba de Wald multiparámetro combinada (D1, Li-Raghunathan-Rubin)
export function poolWald(P, idx) {
  const k = idx.length, m = P.m, Q = idx.map(j => P.beta[j]), Ui = symInv(sub(P.U, idx)); if (!Ui) return { stat: NaN, p: NaN };
  const Bs = sub(P.B, idx); let tr = 0; for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) tr += Bs[a][b] * Ui[b][a];
  const r1 = (1 + 1 / m) * tr / k, D1 = quadForm(Q, Ui) / (k * (1 + r1)), t = k * (m - 1);
  const v1 = t > 4 ? 4 + (t - 4) * (1 + (1 - 2 / t) / r1) ** 2 : t * (1 + 1 / k) * (1 + r1) ** 2 / 2;
  return { stat: D1, df: k, df2: v1, p: pfU(D1, k, v1), F: true };
}
export const pooledCI = (P, j, level = 0.95) => { const q = qt(1 - (1 - level) / 2, P.dfj[j]); return [P.beta[j] - q * P.se[j], P.beta[j] + q * P.se[j]]; };
