// Motores de estimación: MCO, MLG (logística, Poisson), Cox (Efron), Kaplan-Meier, log-rank, GEE y modelo lineal mixto.
import { crossW, crossWv, symInv, mv, dot, logdet, zeros, sub, quadForm } from './linalg.js?v=2026.10.26-1';
import { lgamma, qt, qnorm, p2t, p2z, pchisqU, pfU } from './dist.js?v=2026.10.26-1';

const sum = a => { let s = 0; for (const v of a) s += v; return s; };

// ---------------- regresión lineal (MCO) ----------------
export function fitOLS(X, y, { robust = 'none' } = {}) {
  const n = X.length, p = X[0].length, XtX = crossW(X), XtXi = symInv(XtX);
  if (!XtXi) throw new Error('Matriz de diseño singular (colinealidad perfecta o categorías vacías).');
  const beta = mv(XtXi, crossWv(X, null, y)), fitted = X.map(x => dot(x, beta)), resid = Float64Array.from(y, (v, i) => v - fitted[i]);
  const rss = sum(resid.map(r => r * r)), df = n - p, sigma2 = rss / df;
  const hat = X.map(x => quadForm(x, XtXi));
  let V = XtXi.map(r => r.map(v => v * sigma2));
  if (robust !== 'none') { // HC0, HC1 o HC3
    const meat = crossW(X, Array.from(resid, (r, i) => r * r * (robust === 'HC3' ? 1 / (1 - hat[i]) ** 2 : robust === 'HC1' ? n / df : 1)));
    V = mulABA(XtXi, meat);
  }
  const ym = sum(y) / n, tss = sum(Array.from(y, v => (v - ym) ** 2)), r2 = 1 - rss / tss, k = p - 1;
  const ll = -n / 2 * (Math.log(2 * Math.PI) + Math.log(rss / n) + 1);
  const F = k > 0 ? ((tss - rss) / k) / sigma2 : NaN;
  const cook = Array.from(resid, (r, i) => (r * r / (p * sigma2)) * hat[i] / (1 - hat[i]) ** 2);
  const rstd = Array.from(resid, (r, i) => r / Math.sqrt(sigma2 * (1 - hat[i])));
  return finish({ model: 'ols', beta, V, df, n, ll, aic: -2 * ll + 2 * (p + 1), bic: -2 * ll + Math.log(n) * (p + 1), fitted, resid, hat, cook, rstd, sigma: Math.sqrt(sigma2), r2, adjR2: 1 - (1 - r2) * (n - 1) / df, F, Fp: k > 0 ? pfU(F, k, df) : NaN, Fdf: [k, df], robust, XtXi });
}
function mulABA(A, B) { // A B A (A simétrica)
  const p = A.length, AB = zeros(p, p), o = zeros(p, p);
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) { let s = 0; for (let k = 0; k < p; k++) s += A[i][k] * B[k][j]; AB[i][j] = s; }
  for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) { let s = 0; for (let k = 0; k < p; k++) s += AB[i][k] * A[k][j]; o[i][j] = s; }
  return o;
}

// ---------------- modelos lineales generalizados (enlace canónico) ----------------
const FAM = {
  binomial: { link: e => 1 / (1 + Math.exp(-e)), varf: m => m * (1 - m), ll: (y, m) => (y ? Math.log(Math.max(m, 1e-300)) : Math.log(Math.max(1 - m, 1e-300))), dev: (y, m) => 2 * (y ? -Math.log(Math.max(m, 1e-300)) : -Math.log(Math.max(1 - m, 1e-300))), start: y => Math.log((y + 0.5) / (1.5 - y)) },
  poisson: { link: e => Math.exp(e), varf: m => m, ll: (y, m) => y * Math.log(Math.max(m, 1e-300)) - m - lgamma(y + 1), dev: (y, m) => 2 * ((y > 0 ? y * Math.log(y / m) : 0) - (y - m)), start: y => Math.log(y + 0.1) },
  gaussian: { link: e => e, varf: () => 1, ll: () => 0, dev: (y, m) => (y - m) ** 2, start: y => y },
};
export function fitGLM(X, y, family, { offset = null, robust = false, maxit = 50, ridge = 0 } = {}) {
  const F = FAM[family], n = X.length, p = X[0].length, off = offset || new Float64Array(n);
  let eta = Float64Array.from(y, v => F.start(v)), beta = null, mu, dev = Infinity, conv = false, it = 0;
  // primera iteración desde mu inicial
  for (it = 0; it < maxit; it++) {
    mu = Float64Array.from(eta, (e, i) => F.link(e));
    const w = Float64Array.from(mu, m => Math.max(F.varf(m), 1e-10));
    const z = Float64Array.from(eta, (e, i) => e - off[i] + (y[i] - mu[i]) / w[i]);
    const XtWX = crossW(X, w); if (ridge) for (let j = 1; j < p; j++) XtWX[j][j] += ridge;
    const Hi = symInv(XtWX); if (!Hi) throw new Error('Matriz de información singular (colinealidad, categoría sin eventos o separación).');
    let nb = mv(Hi, crossWv(X, w, z)), ndev;
    // reducción de paso si la devianza aumenta
    for (let h = 0; h < 30; h++) {
      const ne = X.map((x, i) => dot(x, nb) + off[i]);
      ndev = sum(ne.map((e, i) => F.dev(y[i], F.link(e))));
      if (!beta || ndev <= dev + 1e-8 || !isFinite(dev)) { eta = Float64Array.from(ne); break; }
      nb = nb.map((b, j) => (b + beta[j]) / 2);
    }
    const done = beta && Math.abs(ndev - dev) / (Math.abs(ndev) + 0.1) < 1e-10;
    beta = nb; dev = ndev; if (done) { conv = true; break; }
  }
  mu = Float64Array.from(eta, e => F.link(e));
  const w = Float64Array.from(mu, m => Math.max(F.varf(m), 1e-10)), H = crossW(X, w);
  const Hi = symInv(H); let V = Hi;
  const resid = Float64Array.from(y, (v, i) => v - mu[i]);
  if (robust) V = mulABA(Hi, crossW(X, Array.from(resid, r => r * r)));
  const ll = sum(Array.from(y, (v, i) => F.ll(v, mu[i])));
  // modelo nulo (solo intercepto, con offset)
  const ll0 = nullLL(y, family, off);
  const hat = X.map((x, i) => w[i] * quadForm(x, Hi));
  const pres = Float64Array.from(resid, (r, i) => r / Math.sqrt(w[i]));
  const cook = Array.from(pres, (r, i) => (r * r / p) * hat[i] / (1 - hat[i]) ** 2);
  const sep = !conv || beta.some(b => Math.abs(b) > 15) || (family === 'binomial' && Hi.some((r, j) => r[j] > 1e4));
  return finish({ model: family, beta, V, Vnaive: Hi, df: Infinity, n, ll, ll0, dev, aic: -2 * ll + 2 * p, bic: -2 * ll + Math.log(n) * p, fitted: mu, resid, pres, hat, cook, conv, sep, robust, iter: it, eta });
}
function nullLL(y, family, off) {
  const F = FAM[family], n = y.length; let b = 0;
  if (off.every(v => v === 0)) { const m = sum(y) / n; return sum(Array.from(y, v => F.ll(v, m))); }
  for (let it = 0; it < 50; it++) { let s = 0, h = 0; for (let i = 0; i < n; i++) { const m = F.link(b + off[i]); s += y[i] - m; h += F.varf(m); } const d = s / h; b += d; if (Math.abs(d) < 1e-12) break; }
  return sum(Array.from(y, (v, i) => F.ll(v, F.link(b + off[i]))));
}

// ---------------- Cox (empates de Efron) ----------------
export function fitCox(X, time, status, { maxit = 50, ridge = 0 } = {}) {
  const n = X.length, p = X[0].length, ord = Array.from({ length: n }, (_, i) => i).sort((a, b) => time[b] - time[a] || status[a] - status[b]);
  let beta = new Float64Array(p), ll = -Infinity, U, I, conv = false, it;
  // ridge: penaliza ½·λ·β′β (las covariables deben estar estandarizadas); ll, U e I incluyen la penalización
  const evalAt = b => { const c = coxLik(X, time, status, ord, b); if (ridge) { let pen = 0; for (let j = 0; j < p; j++) { pen += b[j] * b[j]; c.U[j] -= ridge * b[j]; c.I[j][j] += ridge; } c.ll -= 0.5 * ridge * pen; } return c; };
  let cur = evalAt(beta); const ll0 = cur.ll;
  for (it = 0; it < maxit; it++) {
    const Ii = symInv(cur.I); if (!Ii) throw new Error('Matriz de información singular en Cox (covariables colineales o sin eventos).');
    let step = mv(Ii, cur.U), nb, nx;
    for (let h = 0; h < 30; h++) { nb = beta.map((v, j) => v + step[j]); nx = evalAt(nb); if (nx.ll >= cur.ll - 1e-10) break; step = step.map(s => s / 2); }
    const done = Math.abs(nx.ll - cur.ll) < 1e-10 * (1 + Math.abs(cur.ll));
    beta = Float64Array.from(nb); cur = nx; if (done) { conv = true; break; }
  }
  ({ ll, U, I } = cur);
  const V = symInv(I), nev = sum(status);
  const s0 = evalAt(new Float64Array(p)), Ii0 = symInv(s0.I), score = Ii0 ? quadForm(s0.U, Ii0) : NaN;
  const eta = X.map(x => dot(x, beta));
  return finish({ model: 'cox', beta, V, df: Infinity, n, nevent: nev, ll, ll0, aic: -2 * ll + 2 * p, bic: -2 * ll + Math.log(nev) * p, conv, iter: it, eta, ord, score, scoreP: pchisqU(score, p), lr: 2 * (ll - ll0), lrP: pchisqU(2 * (ll - ll0), p), sep: beta.some(b => Math.abs(b) > 15) });
}
// log-verosimilitud parcial (Efron) de un β dado
export function coxLogLik(X, time, status, beta) { const n = X.length, ord = Array.from({ length: n }, (_, i) => i).sort((a, b) => time[b] - time[a] || status[a] - status[b]); return coxLik(X, time, status, ord, beta).ll; }
function coxLik(X, time, status, ord, beta) {
  const n = X.length, p = beta.length, r = new Float64Array(n);
  for (let i = 0; i < n; i++) r[i] = Math.exp(dot(X[i], beta));
  let ll = 0; const U = new Float64Array(p), I = zeros(p, p);
  let S0 = 0; const S1 = new Float64Array(p), S2 = zeros(p, p);
  for (let k = 0; k < n;) {
    // bloque de tiempos iguales (orden descendente)
    let j = k; while (j + 1 < n && time[ord[j + 1]] === time[ord[k]]) j++;
    let D0 = 0; const D1 = new Float64Array(p), D2 = zeros(p, p); let d = 0;
    for (let m = k; m <= j; m++) {
      const i = ord[m], x = X[i], ri = r[i];
      S0 += ri; for (let a = 0; a < p; a++) { S1[a] += ri * x[a]; for (let b = 0; b < p; b++) S2[a][b] += ri * x[a] * x[b]; }
      if (status[i]) { d++; D0 += ri; ll += Math.log(ri); for (let a = 0; a < p; a++) { U[a] += x[a]; D1[a] += ri * x[a]; for (let b = 0; b < p; b++) D2[a][b] += ri * x[a] * x[b]; } }
    }
    for (let l = 0; l < d; l++) {
      const f = l / d, s0 = S0 - f * D0;
      ll -= Math.log(s0);
      const s1 = Float64Array.from(S1, (v, a) => v - f * D1[a]);
      for (let a = 0; a < p; a++) { U[a] -= s1[a] / s0; for (let b = 0; b < p; b++) I[a][b] += (S2[a][b] - f * D2[a][b]) / s0 - s1[a] * s1[b] / (s0 * s0); }
    }
    k = j + 1;
  }
  return { ll, U, I, r };
}
// residuos de Schoenfeld (promedio de Efron en empates) y prueba de riesgos proporcionales (Grambsch-Therneau)
export function coxZph(fit, X, time, status, transform = 'km') {
  const n = X.length, p = X[0].length, { ord, beta } = fit, r = X.map(x => Math.exp(dot(x, beta)));
  const res = [], times = [];
  let S0 = 0; const S1 = new Float64Array(p);
  for (let k = 0; k < n;) {
    let j = k; while (j + 1 < n && time[ord[j + 1]] === time[ord[k]]) j++;
    let D0 = 0; const D1 = new Float64Array(p); const dead = [];
    for (let m = k; m <= j; m++) { const i = ord[m]; S0 += r[i]; for (let a = 0; a < p; a++) S1[a] += r[i] * X[i][a]; if (status[i]) { dead.push(i); D0 += r[i]; for (let a = 0; a < p; a++) D1[a] += r[i] * X[i][a]; } }
    const d = dead.length;
    if (d) {
      const xbar = new Float64Array(p);
      for (let l = 0; l < d; l++) { const f = l / d, s0 = S0 - f * D0; for (let a = 0; a < p; a++) xbar[a] += (S1[a] - f * D1[a]) / s0 / d; }
      for (const i of dead) { res.push(Float64Array.from(X[i], (v, a) => v - xbar[a])); times.push(time[i]); }
    }
    k = j + 1;
  }
  // orden ascendente de tiempo
  const o = times.map((_, i) => i).sort((a, b) => times[a] - times[b]);
  const R = o.map(i => res[i]), tt = o.map(i => times[i]), d = R.length;
  let g;
  if (transform === 'identity') g = tt;
  else if (transform === 'log') g = tt.map(Math.log);
  else if (transform === 'rank') g = tt.map((t, i) => i + 1);
  else { const k = km(time, status); g = tt.map(t => 1 - survLeft(k, t)); }
  const gm = sum(g) / d, gc = g.map(v => v - gm), sgg = sum(gc.map(v => v * v));
  const Vb = fit.V, Uv = new Float64Array(p);
  R.forEach((s, k) => { for (let a = 0; a < p; a++) Uv[a] += gc[k] * s[a]; });
  const per = Array.from({ length: p }, (_, a) => { const chisq = d * Uv[a] * Uv[a] * Vb[a][a] / sgg; return { chisq, df: 1, p: pchisqU(chisq, 1) }; });
  const gl = d * quadForm(Uv, Vb) / sgg;
  // residuos escalados para el gráfico: beta + d · V · s_k
  const scaled = R.map(s => Float64Array.from(mv(Vb, s), (v, a) => beta[a] + d * v));
  return { per, global: { chisq: gl, df: p, p: pchisqU(gl, p) }, times: tt, g, scaled, U: Uv, d, sgg, term: idx => { const c = d * quadForm(idx.map(j => Uv[j]), sub(Vb, idx)) / sgg; return { chisq: c, df: idx.length, p: pchisqU(c, idx.length) }; } };
}
// supervivencia justo antes de t
function survLeft(k, t) { let s = 1; for (const st of k.steps) { if (st.t >= t) break; s = st.surv; } return s; }

// índice C de Harrell (pares comparables, empates en predictor = 1/2)
export function concordance(eta, time, status) {
  const n = eta.length; if (n > 6000) return NaN;
  let conc = 0, tot = 0;
  for (let i = 0; i < n; i++) if (status[i]) for (let j = 0; j < n; j++) if (time[j] > time[i]) { tot++; if (eta[i] > eta[j]) conc++; else if (eta[i] === eta[j]) conc += 0.5; }
  return conc / tot;
}

// ---------------- Kaplan-Meier y log-rank ----------------
export function km(time, status, level = 0.95) {
  const z = qnorm(1 - (1 - level) / 2), idx = Array.from(time.keys()).sort((a, b) => time[a] - time[b]);
  const n = idx.length; let atRisk = n, s = 1, gw = 0; const steps = [];
  for (let k = 0; k < n;) {
    let j = k, d = 0, c = 0; while (j < n && time[idx[j]] === time[idx[k]]) { if (status[idx[j]]) d++; else c++; j++; }
    if (d > 0) { s *= 1 - d / atRisk; gw += atRisk > d ? d / (atRisk * (atRisk - d)) : Infinity; }
    const se = s * Math.sqrt(gw);
    // IC en escala log (conf.type = "log" de R)
    const lo = s > 0 ? Math.exp(Math.log(s) - z * Math.sqrt(gw)) : 0, hi = s > 0 ? Math.min(1, Math.exp(Math.log(s) + z * Math.sqrt(gw))) : 0;
    steps.push({ t: time[idx[k]], nrisk: atRisk, nevent: d, ncens: c, surv: s, se, lo, hi });
    atRisk -= d + c; k = j;
  }
  const medianOf = key => { const st = steps.find(x => x[key] <= 0.5 + 1e-12 && x.nevent >= 0 && x[key] < 0.5 + 1e-12); return st ? st.t : NaN; };
  return { steps, n, events: steps.reduce((a, b) => a + b.nevent, 0), median: medianOf('surv'), medianCI: [medianOf('lo'), medianOf('hi')] };
}
export function logrank(time, status, group) {
  const lv = [...new Set(group)], K = lv.length, gi = group.map(g => lv.indexOf(g));
  const ts = [...new Set(Array.from(time).filter((t, i) => status[i]))].sort((a, b) => a - b);
  const O = new Float64Array(K), E = new Float64Array(K), Vm = zeros(K, K);
  for (const t of ts) {
    const nk = new Float64Array(K), dk = new Float64Array(K);
    for (let i = 0; i < time.length; i++) if (time[i] >= t) { nk[gi[i]]++; if (time[i] === t && status[i]) dk[gi[i]]++; }
    const N = sum(nk), D = sum(dk); if (N < 1) continue;
    for (let a = 0; a < K; a++) {
      O[a] += dk[a]; E[a] += D * nk[a] / N;
      if (N > 1) for (let b = 0; b < K; b++) Vm[a][b] += D * (N - D) / (N - 1) * (nk[a] / N) * ((a === b ? 1 : 0) - nk[b] / N);
    }
  }
  const idx = Array.from({ length: K - 1 }, (_, i) => i), d = idx.map(i => O[i] - E[i]), Vi = symInv(sub(Vm, idx));
  const chisq = Vi ? quadForm(d, Vi) : NaN;
  return { levels: lv, O: Array.from(O), E: Array.from(E), chisq, df: K - 1, p: pchisqU(chisq, K - 1) };
}

// ---------------- GEE (Liang-Zeger) ----------------
export function fitGEE(X, y, cluster, family = 'gaussian', corstr = 'exchangeable', { maxit = 100 } = {}) {
  const F = FAM[family], n = X.length, p = X[0].length;
  const groups = new Map(); cluster.forEach((c, i) => { if (!groups.has(c)) groups.set(c, []); groups.get(c).push(i); });
  const G = [...groups.values()];
  let beta = Float64Array.from(fitGLM(X, y, family === 'gaussian' ? 'gaussian' : family).beta), alpha = 0, phi = 1;
  const corInv = (m, a) => { const R = zeros(m, m); for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) R[i][j] = i === j ? 1 : corstr === 'exchangeable' ? a : corstr === 'ar1' ? Math.pow(a, Math.abs(i - j)) : 0; return symInv(R); };
  const state = b => { const mu = X.map(x => F.link(dot(x, b))), v = mu.map(m => Math.max(F.varf(m), 1e-10)); return { mu, v, r: y.map ? Array.from(y, (yy, i) => (yy - mu[i]) / Math.sqrt(v[i])) : null }; };
  const updateDep = (st) => {
    phi = sum(st.r.map(r => r * r)) / (n - p);
    if (corstr === 'independence') { alpha = 0; return; }
    let s = 0, np = 0;
    for (const g of G) {
      if (corstr === 'exchangeable') { let a = 0, b = 0; for (const i of g) { a += st.r[i]; b += st.r[i] ** 2; } s += (a * a - b) / 2; np += g.length * (g.length - 1) / 2; }
      else for (let k = 0; k + 1 < g.length; k++) { s += st.r[g[k]] * st.r[g[k + 1]]; np++; }
    }
    alpha = Math.max(-0.99, Math.min(0.99, s / phi / (np - p)));
  };
  const accumulate = (b, withMeat) => {
    const st = state(b), H = zeros(p, p), U = new Float64Array(p), M = zeros(p, p), cache = {};
    for (const g of G) {
      const m = g.length, Ri = cache[m] || (cache[m] = corInv(m, alpha));
      // D = diag(dmu/deta) X ; V = A^1/2 R A^1/2 ; enlace canónico: dmu/deta = v
      const D = g.map(i => Float64Array.from(X[i], x => x * st.v[i])), sv = g.map(i => Math.sqrt(st.v[i]));
      const Vi = Ri.map((row, a) => row.map((val, c) => val / (sv[a] * sv[c])));
      const res = g.map(i => y[i] - st.mu[i]);
      const DtVi = zeros(p, m);
      for (let a = 0; a < p; a++) for (let c = 0; c < m; c++) { let s = 0; for (let k = 0; k < m; k++) s += D[k][a] * Vi[k][c]; DtVi[a][c] = s; }
      const u = new Float64Array(p);
      for (let a = 0; a < p; a++) { for (let c = 0; c < m; c++) u[a] += DtVi[a][c] * res[c]; U[a] += u[a]; for (let b2 = 0; b2 < p; b2++) { let s = 0; for (let c = 0; c < m; c++) s += DtVi[a][c] * D[c][b2]; H[a][b2] += s; } }
      if (withMeat) for (let a = 0; a < p; a++) for (let b2 = 0; b2 < p; b2++) M[a][b2] += u[a] * u[b2];
    }
    return { st, H, U, M };
  };
  let conv = false, it;
  updateDep(state(beta));
  for (it = 0; it < maxit; it++) {
    const { H, U } = accumulate(beta, false), Hi = symInv(H); if (!Hi) throw new Error('GEE: matriz singular.');
    const step = mv(Hi, U); beta = beta.map((b, j) => b + step[j]);
    updateDep(state(beta));
    if (Math.max(...step.map(Math.abs)) < 1e-9) { conv = true; break; }
  }
  const { st, H, M } = accumulate(beta, true), Hi = symInv(H), V = mulABA(Hi, M);
  const sc = family === 'gaussian' ? phi : 1, Vnaive = Hi.map(r => r.map(v => v * sc));
  // QIC (Pan 2001): cuasi-verosimilitud bajo independencia + penalización traza(Omega_I · V_R)
  const w = st.v, OmegaI = crossW(X, Array.from(w, v => v / sc));
  let tr = 0; for (let a = 0; a < p; a++) for (let b2 = 0; b2 < p; b2++) tr += OmegaI[a][b2] * V[b2][a];
  const quasi = sum(Array.from(y, (yy, i) => family === 'gaussian' ? -((yy - st.mu[i]) ** 2) / 2 / sc : family === 'binomial' ? (yy * Math.log(st.mu[i] / (1 - st.mu[i])) + Math.log(1 - st.mu[i])) : yy * Math.log(st.mu[i]) - st.mu[i]));
  return finish({ model: 'gee', family, corstr, beta, V, Vnaive, df: Infinity, n, nclust: G.length, alpha, phi, conv, iter: it, qic: -2 * quasi + 2 * tr, qicu: -2 * quasi + 2 * p, fitted: st.mu, resid: Array.from(y, (yy, i) => yy - st.mu[i]) });
}

// ---------------- modelo lineal mixto con intercepto aleatorio (REML) ----------------
export function fitLMM(X, y, cluster) {
  const n = X.length, p = X[0].length, groups = new Map();
  cluster.forEach((c, i) => { if (!groups.has(c)) groups.set(c, []); groups.get(c).push(i); });
  const G = [...groups.values()];
  const at = gam => {
    const H = zeros(p, p), b = new Float64Array(p);
    for (const g of G) {
      const c = gam / (1 + g.length * gam), sx = new Float64Array(p); let sy = 0;
      for (const i of g) { sy += y[i]; for (let a = 0; a < p; a++) sx[a] += X[i][a]; }
      for (const i of g) for (let a = 0; a < p; a++) { b[a] += X[i][a] * y[i]; for (let k = 0; k < p; k++) H[a][k] += X[i][a] * X[i][k]; }
      for (let a = 0; a < p; a++) { b[a] -= c * sx[a] * sy; for (let k = 0; k < p; k++) H[a][k] -= c * sx[a] * sx[k]; }
    }
    const Hi = symInv(H); if (!Hi) throw new Error('Modelo mixto: matriz singular.');
    const beta = mv(Hi, b); let rss = 0;
    for (const g of G) { const c = gam / (1 + g.length * gam); let s = 0, ss = 0; for (const i of g) { const e = y[i] - dot(X[i], beta); s += e; ss += e * e; } rss += ss - c * s * s; }
    const crit = (n - p) * Math.log(rss / (n - p)) + sum(G.map(g => Math.log(1 + g.length * gam))) + logdet(H);
    return { beta, Hi, rss, crit };
  };
  // búsqueda en sección áurea sobre log(gamma), gamma = sigma_b^2 / sigma^2
  let a = -12, b = 6; const gr = (Math.sqrt(5) - 1) / 2; let c = b - gr * (b - a), d = a + gr * (b - a), fc = at(Math.exp(c)).crit, fd = at(Math.exp(d)).crit;
  for (let i = 0; i < 120; i++) { if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = at(Math.exp(c)).crit; } else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = at(Math.exp(d)).crit; } }
  let gam = Math.exp((a + b) / 2), best = at(gam);
  const zero = at(0); if (zero.crit <= best.crit) { gam = 0; best = zero; }
  const s2 = best.rss / (n - p), V = best.Hi.map(r => r.map(v => v * s2));
  const llR = -0.5 * (best.crit + (n - p) + (n - p) * Math.log(2 * Math.PI));
  // LRT del efecto aleatorio (mezcla 50:50 de chi² 0 y 1)
  const llR0 = -0.5 * (zero.crit + (n - p) + (n - p) * Math.log(2 * Math.PI)), lrt = Math.max(0, 2 * (llR - llR0));
  const fitted = X.map(x => dot(x, best.beta)), cond = new Float64Array(n), blup = new Map();
  // BLUP del intercepto aleatorio y residuos condicionales (nivel 1)
  G.forEach(g => { const c = gam / (1 + g.length * gam); let s = 0; for (const i of g) s += y[i] - fitted[i]; const b = c * s; g.forEach(i => { cond[i] = y[i] - fitted[i] - b; }); blup.set(cluster[g[0]], b); });
  return finish({ model: 'lmm', cond, blup, beta: best.beta, V, df: Infinity, n, nclust: G.length, sigma2: s2, tau2: gam * s2, icc: gam / (1 + gam), llREML: llR, aic: -2 * llR + 2 * (p + 2), lrtRE: lrt, lrtREp: 0.5 * pchisqU(lrt, 1), fitted, resid: Array.from(y, (v, i) => v - fitted[i]) });
}

// ---------------- utilidades comunes ----------------
function finish(f) { f.se = f.V.map((r, j) => Math.sqrt(r[j])); f.stat = Array.from(f.beta, (b, j) => b / f.se[j]); f.p = f.stat.map(z => (isFinite(f.df) ? p2t(z, f.df) : p2z(z))); return f; }
export function ciOf(f, j, level = 0.95) { const q = isFinite(f.df) ? qt(1 - (1 - level) / 2, f.df) : qnorm(1 - (1 - level) / 2); return [f.beta[j] - q * f.se[j], f.beta[j] + q * f.se[j]]; }
// prueba de Wald conjunta para un término (varias columnas); F si el modelo tiene gl residuales
export function waldTerm(f, idx) {
  const b = idx.map(j => f.beta[j]), Vi = symInv(sub(f.V, idx)); if (!Vi) return { stat: NaN, df: idx.length, p: NaN };
  const w = quadForm(b, Vi);
  return isFinite(f.df) ? { stat: w / idx.length, df: idx.length, df2: f.df, p: pfU(w / idx.length, idx.length, f.df), F: true } : { stat: w, df: idx.length, p: pchisqU(w, idx.length) };
}
