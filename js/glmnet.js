// Regresión penalizada (red elástica) por descenso de coordenadas, al estilo de glmnet (Friedman, Hastie y Tibshirani, 2010).
// Familias: gaussiana y binomial (logística). α = 1 lasso, α = 0 ridge. Las covariables se estandarizan internamente
// y los coeficientes se devuelven en la escala original. El intercepto no se penaliza.
import { rng } from './dist.js?v=2026.10.25-8';

const sgn = (z, g) => (z > g ? z - g : z < -g ? z + g : 0);
const expit = e => 1 / (1 + Math.exp(-e));

export function standardize(X) {
  const n = X.length, p = X[0].length, mean = new Float64Array(p), sd = new Float64Array(p);
  for (const x of X) for (let j = 0; j < p; j++) mean[j] += x[j] / n;
  for (const x of X) for (let j = 0; j < p; j++) sd[j] += (x[j] - mean[j]) ** 2 / n;
  for (let j = 0; j < p; j++) sd[j] = Math.sqrt(sd[j]) || 1;
  return { mean, sd, Z: X.map(x => Float64Array.from(x, (v, j) => (v - mean[j]) / sd[j])) };
}

// ajuste en la escala estandarizada; devuelve {b0, b, iters}
function fitStd(Z, y, family, alpha, lambda, { offset = null, start = null, pf = null, maxit = 200, tol = 1e-8 } = {}) {
  const n = Z.length, p = Z[0].length, off = offset || new Float64Array(n);
  const b = start ? Float64Array.from(start.b) : new Float64Array(p); let b0 = start ? start.b0 : 0;
  if (!start && family === 'binomial') { const m = y.reduce((s, v) => s + v, 0) / n; b0 = Math.log(Math.max(m, 1e-6) / Math.max(1 - m, 1e-6)); }
  if (!start && family === 'gaussian') b0 = y.reduce((s, v, i) => s + v - off[i], 0) / n;
  const w = new Float64Array(n).fill(1), z = new Float64Array(n), r = new Float64Array(n), eta = new Float64Array(n);
  let outer = 0;
  for (; outer < (family === 'binomial' ? 50 : 1); outer++) {
    for (let i = 0; i < n; i++) { let e = b0 + off[i]; const x = Z[i]; for (let j = 0; j < p; j++) if (b[j] !== 0) e += x[j] * b[j]; eta[i] = e; }
    if (family === 'binomial') for (let i = 0; i < n; i++) { const mu = Math.min(Math.max(expit(eta[i]), 1e-5), 1 - 1e-5), wi = mu * (1 - mu); w[i] = wi; z[i] = eta[i] - off[i] + (y[i] - mu) / wi; }
    else for (let i = 0; i < n; i++) z[i] = y[i] - off[i];
    // residuo de trabajo r = z − b0 − Zb
    for (let i = 0; i < n; i++) r[i] = z[i] - (eta[i] - off[i]);
    const prev = Float64Array.from(b), prev0 = b0;
    for (let it = 0; it < maxit; it++) {
      let maxd = 0;
      // intercepto
      { let sw = 0, sr = 0; for (let i = 0; i < n; i++) { sw += w[i]; sr += w[i] * r[i]; } const d = sr / sw; b0 += d; for (let i = 0; i < n; i++) r[i] -= d; maxd = Math.max(maxd, Math.abs(d)); }
      for (let j = 0; j < p; j++) {
        let g = 0, v = 0; for (let i = 0; i < n; i++) { const xi = Z[i][j]; g += w[i] * xi * r[i]; v += w[i] * xi * xi; }
        g /= n; v /= n; const pj = pf ? pf[j] : 1, nb = sgn(g + v * b[j], lambda * alpha * pj) / (v + lambda * (1 - alpha) * pj);
        const d = nb - b[j]; if (d !== 0) { for (let i = 0; i < n; i++) r[i] -= d * Z[i][j]; b[j] = nb; maxd = Math.max(maxd, Math.abs(d) * Math.sqrt(v)); }
      }
      if (maxd < tol) break;
    }
    let ch = Math.abs(b0 - prev0); for (let j = 0; j < p; j++) ch = Math.max(ch, Math.abs(b[j] - prev[j]));
    if (ch < 1e-7) break;
  }
  return { b0, b, iters: outer };
}
const back = (fit, st) => { const beta = Float64Array.from(fit.b, (v, j) => v / st.sd[j]); let b0 = fit.b0; for (let j = 0; j < beta.length; j++) b0 -= beta[j] * st.mean[j]; return { intercept: b0, beta }; };

// λ máximo (todos los coeficientes en cero)
export function lambdaMax(Z, y, family, alpha, pf = null) {
  const n = Z.length, p = Z[0].length; let mu;
  if (family === 'binomial') { const m = y.reduce((s, v) => s + v, 0) / n; mu = new Float64Array(n).fill(m); } else { const m = y.reduce((s, v) => s + v, 0) / n; mu = new Float64Array(n).fill(m); }
  let mx = 0; for (let j = 0; j < p; j++) { let g = 0; for (let i = 0; i < n; i++) g += Z[i][j] * (y[i] - mu[i]); mx = Math.max(mx, Math.abs(g) / n / (pf ? pf[j] || 1e-9 : 1)); }
  return mx / Math.max(alpha, 1e-3);
}

// ajuste con un λ dado (escala original)
export function enet(X, y, { family = 'binomial', alpha = 1, lambda, offset = null, pf = null } = {}) {
  const st = standardize(X), fit = fitStd(st.Z, y, family, alpha, lambda, { offset, pf }), { intercept, beta } = back(fit, st);
  return { intercept, beta, lambda, alpha, family, df: beta.filter(v => v !== 0).length };
}
// trayectoria de λ (de mayor a menor, con arranque en caliente)
export function enetPath(X, y, { family = 'binomial', alpha = 1, nlambda = 50, ratio = null, pf = null } = {}) {
  const st = standardize(X), n = X.length, p = X[0].length, lmax = lambdaMax(st.Z, y, family, alpha, pf), r = ratio ?? (n > p ? 1e-3 : 1e-2);
  const lambdas = Array.from({ length: nlambda }, (_, k) => lmax * Math.pow(r, k / (nlambda - 1))); let start = null; const fits = [];
  for (const lam of lambdas) { const f = fitStd(st.Z, y, family, alpha, lam, { start, pf }); start = f; fits.push({ lambda: lam, ...back(f, st), df: f.b.filter(v => v !== 0).length }); }
  return { lambdas, fits };
}
export const predictLP = (fit, X) => Float64Array.from(X, x => { let e = fit.intercept; for (let j = 0; j < x.length; j++) e += x[j] * fit.beta[j]; return e; });

// devianza media por observación
function devMean(y, lp, family) {
  let s = 0; const n = y.length;
  if (family === 'binomial') for (let i = 0; i < n; i++) { const e = lp[i]; s += (y[i] ? Math.log1p(Math.exp(-e)) : Math.log1p(Math.exp(e))) * 2; } else for (let i = 0; i < n; i++) s += (y[i] - lp[i]) ** 2;
  return s / n;
}
// validación cruzada de λ (K pliegues), con reglas λ_min y λ_1ES
export function cvEnet(X, y, { family = 'binomial', alpha = 1, nfolds = 10, nlambda = 50, seed = 1, pf = null, strata = true } = {}) {
  const n = X.length, R = rng(seed), path = enetPath(X, y, { family, alpha, nlambda, pf }), lam = path.lambdas, K = Math.min(nfolds, n);
  // pliegues (estratificados por el desenlace en el caso binomial)
  const fold = new Int32Array(n), ord = Array.from({ length: n }, (_, i) => i);
  if (family === 'binomial' && strata) ord.sort((a, b) => y[a] - y[b] || R.u() - 0.5); else ord.sort(() => R.u() - 0.5);
  ord.forEach((i, k) => (fold[i] = k % K));
  const errs = Array.from({ length: K }, () => new Float64Array(lam.length));
  for (let k = 0; k < K; k++) {
    const tr = [], te = []; for (let i = 0; i < n; i++) (fold[i] === k ? te : tr).push(i);
    const Xt = tr.map(i => X[i]), yt = Float64Array.from(tr, i => y[i]), st = standardize(Xt); let start = null;
    const Xe = te.map(i => X[i]), ye = Float64Array.from(te, i => y[i]);
    lam.forEach((l, li) => { const f = fitStd(st.Z, yt, family, alpha, l, { start, pf }); start = f; const bk = back(f, st); errs[k][li] = devMean(ye, predictLP(bk, Xe), family); });
  }
  const mean = lam.map((_, li) => errs.reduce((s, e) => s + e[li], 0) / K), se = lam.map((_, li) => { const m = mean[li]; return Math.sqrt(errs.reduce((s, e) => s + (e[li] - m) ** 2, 0) / (K - 1) / K); });
  let imin = 0; mean.forEach((m, i) => { if (m < mean[imin]) imin = i; });
  let i1 = imin; for (let i = 0; i < lam.length; i++) if (mean[i] <= mean[imin] + se[imin]) { i1 = i; break; }
  return { lambdas: lam, cv: mean, se, lambdaMin: lam[imin], lambda1se: lam[i1], imin, i1, path };
}
