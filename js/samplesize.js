// Tamaño de muestra para modelos de predicción clínica.
//  · Desarrollo (Riley et al., Stat Med 2019; BMJ 2020): resultados binarios, continuos y de supervivencia (criterios de pmsampsize).
//  · Validación externa (Riley et al., Stat Med 2021; BMJ 2024;384:e074821): resultados binarios y continuos (criterios de pmvalsampsize).
import { qt, qnorm, pnorm, dnorm } from './dist.js?v=2026.10.25-6';

const ceil = Math.ceil, ln = Math.log, exp = Math.exp, Z = qnorm(0.975);
const expit = x => 1 / (1 + exp(-x));
// n tal que el factor de contracción global de van Houwelingen-Le Cessie sea S:  n = p / ((S − 1) ln(1 − R²/S))
export const nShrink = (p, r2, S) => p / ((S - 1) * ln(1 - r2 / S));
// R² de Cox-Snell máximo posible: 1 − exp(2 lnL_null / n)
export const maxR2binary = prev => 1 - exp(2 * (prev * ln(prev) + (1 - prev) * ln(1 - prev)));
export const maxR2survival = (rate, meanfup, n = 1e4) => { const E = ceil(rate * meanfup * n), l0 = E * ln(E / n) - E; return 1 - exp(2 * l0 / n); };

// R² de Cox-Snell que corresponde a un estadístico C y una prevalencia, suponiendo LP normal con varianza común
// (no eventos N(0,1); eventos N(μ,1), μ = √2 Φ⁻¹(C)); se integra numéricamente en lugar de simular.
export function cstatToR2cs(C, prev) {
  const mu = Math.SQRT2 * qnorm(C), b = mu, a = ln(prev / (1 - prev)) - mu * mu / 2, lo = -9, hi = 9 + mu, N = 6000, h = (hi - lo) / N;
  let l1 = 0;
  for (let i = 0; i <= N; i++) {
    const x = lo + i * h, w = (i === 0 || i === N ? 0.5 : 1) * h, eta = a + b * x;
    const lp1 = -Math.log1p(exp(-eta)), lp0 = -Math.log1p(exp(eta)); // ln p, ln(1−p) estables
    l1 += w * (prev * dnorm(x - mu) * lp1 + (1 - prev) * dnorm(x) * lp0);
  }
  const l0 = prev * ln(prev) + (1 - prev) * ln(1 - prev);
  return 1 - exp(2 * (l0 - l1));
}

// ---------------------------------------------------------------- desarrollo
export function devBinary({ r2cs = null, nag = null, cstat = null, p, prev, shrink = 0.9, delta = 0.05, margin = 0.05 }) {
  const mx = maxR2binary(prev);
  const r2 = r2cs ?? (nag != null ? nag * mx : cstatToR2cs(cstat, prev));
  if (shrink <= r2) throw new Error('El factor de contracción objetivo debe ser mayor que el R² de Cox-Snell.');
  if (r2 > mx) throw new Error(`El R² de Cox-Snell (${r2.toFixed(3)}) supera el máximo posible (${mx.toFixed(3)}) para esa prevalencia.`);
  const n1 = ceil(nShrink(p, r2, shrink)), S2 = r2 / (r2 + delta * mx), n2 = ceil(nShrink(p, r2, S2)), n3 = ceil((Z / margin) ** 2 * prev * (1 - prev));
  return finish('binary', [['Contracción global (S ≥ ' + shrink + ')', n1, shrink], ['Diferencia pequeña entre R² aparente y ajustado (' + delta + ')', n2, S2], ['Precisión del riesgo global (± ' + margin + ')', n3, Math.max(S2, shrink)]], { r2cs: r2, maxR2: mx, nagR2: r2 / mx, p, prev });
}
export function devSurvival({ r2cs = null, nag = null, p, rate, timepoint, meanfup, shrink = 0.9, delta = 0.05, margin = 0.05 }) {
  const mx = maxR2survival(rate, meanfup), r2 = r2cs ?? nag * mx;
  if (shrink <= r2) throw new Error('El factor de contracción objetivo debe ser mayor que el R² de Cox-Snell.');
  if (r2 > mx) throw new Error(`El R² de Cox-Snell (${r2.toFixed(3)}) supera el máximo posible (${mx.toFixed(3)}) para esa tasa y seguimiento.`);
  const n1 = ceil(nShrink(p, r2, shrink)), S2 = r2 / (r2 + delta * mx), n2 = ceil(nShrink(p, r2, S2));
  // criterio 3: el IC 95 % del riesgo acumulado en `timepoint` (modelo exponencial) debe tener un margen ≤ `margin`
  const risk = 1 - exp(-rate * timepoint), marg = n => { const T = meanfup * n; return 1 - exp(-(rate + Z * Math.sqrt(rate / T)) * timepoint) - risk; };
  let n3 = 1; while (marg(n3) > margin && n3 < 1e7) n3 = n3 < 1000 ? n3 + 1 : ceil(n3 * 1.002);
  const r = finish('survival', [['Contracción global (S ≥ ' + shrink + ')', n1, shrink], ['Diferencia pequeña entre R² aparente y ajustado (' + delta + ')', n2, S2], ['Precisión del riesgo acumulado a ' + timepoint + ' (± ' + margin + ')', n3, Math.max(S2, shrink)]], { r2cs: r2, maxR2: mx, nagR2: r2 / mx, p, rate, meanfup, timepoint });
  r.events = r.n * rate * meanfup; r.epp = r.events / p; r.personTime = r.n * meanfup;
  r.crit.forEach(c => { c.events = c.n * rate * meanfup; c.epp = c.events / p; });
  return r;
}
export function devContinuous({ r2, p, intercept, sd, shrink = 0.9, mmoe = 1.1, delta = 0.05 }) {
  const es = (n) => 1 + (p - 2) / (n * ln(1 - (r2 * (n - p - 1) + p) / (n - 1)));
  let n1 = p + 2; while (!(es(n1) >= shrink) && n1 < 5e7) n1 += n1 < 5000 ? 1 : ceil(n1 * 0.001);
  const n2 = ceil(1 + p * (1 - r2) / delta), n3 = 234 + p, n0 = Math.max(n1, n2, n3);
  let n4 = n0;
  if (isFinite(intercept) && intercept > 0 && isFinite(sd)) { const ratio = n => (intercept + qt(0.975, n - p - 1) * Math.sqrt(sd * sd * (1 - r2) / n)) / intercept; while (ratio(n4) > mmoe && n4 < 5e7) n4++; }
  const r = finish('continuous', [['Contracción global (S ≥ ' + shrink + ')', n1, es(n1)], ['Diferencia pequeña entre R² aparente y ajustado (' + delta + ')', n2, es(n2)], ['Precisión de la varianza residual (MMOE ≤ ' + mmoe + ')', n3, es(n3)], ['Precisión del intercepto (media del desenlace)', n4, es(n4)]], { r2, p });
  return r;
}
function finish(type, rows, extra) {
  const crit = rows.map(([label, n, shrinkage]) => ({ label, n: ceil(n), shrinkage })), n = Math.max(...crit.map(c => c.n));
  const out = { type, crit, n, ...extra };
  if (type === 'binary') { crit.forEach(c => { c.events = c.n * extra.prev; c.epp = c.events / extra.p; }); out.events = n * extra.prev; out.epp = out.events / extra.p; }
  if (type === 'continuous') { crit.forEach(c => (c.spp = c.n / extra.p)); out.spp = n / extra.p; }
  return out;
}

// ---------------------------------------------------------------- validación externa
// Distribución del predictor lineal (LP) bajo calibración perfecta (α = 0, β = 1):
//  · desde C y prevalencia: LP|evento ~ N(μ0+Δ, Δ), LP|no evento ~ N(μ0, Δ) con Δ = 2 Φ⁻¹(C)², μ0 = logit(φ) − Δ/2
//  · desde LP ~ N(m, s²): la prevalencia implícita es E[expit(LP)]
export function lpDistribution({ prev, cstat, lpMean, lpSd }) {
  const grid = [];
  if (isFinite(lpMean) && isFinite(lpSd) && lpSd > 0) {
    const lo = lpMean - 9 * lpSd, hi = lpMean + 9 * lpSd, N = 5000, h = (hi - lo) / N;
    for (let i = 0; i <= N; i++) { const x = lo + i * h, f = dnorm((x - lpMean) / lpSd) / lpSd * (i === 0 || i === N ? 0.5 : 1) * h; grid.push({ x, f, p: expit(x) }); }
  } else {
    const D = 2 * qnorm(cstat) ** 2, mu0 = ln(prev / (1 - prev)) - D / 2, sd = Math.sqrt(D), lo = mu0 - 9 * sd, hi = mu0 + D + 9 * sd, N = 6000, h = (hi - lo) / N;
    for (let i = 0; i <= N; i++) { const x = lo + i * h, w = (i === 0 || i === N ? 0.5 : 1) * h, f = (prev * dnorm((x - mu0 - D) / sd) + (1 - prev) * dnorm((x - mu0) / sd)) / sd * w; grid.push({ x, f, p: expit(x) }); }
  }
  const phi = grid.reduce((s, g) => s + g.f * g.p, 0);
  return { grid, phi };
}
export const seC = (C, phi, N) => Math.sqrt(C * (1 - C) * (1 + (N / 2 - 1) * ((1 - C) / (2 - C) + C / (1 + C))) / (N * N * phi * (1 - phi)));
export function valBinary({ prev, cstat, lpMean = NaN, lpSd = NaN, seOE = 0.051, seSlope = 0.051, seCstat = 0.0255, thresholds = [], seSNB = null }) {
  const { grid, phi } = lpDistribution({ prev, cstat, lpMean, lpSd }), C = isFinite(cstat) ? cstat : cFromGrid(grid, phi);
  let Ia = 0, Ib = 0, Iab = 0;
  for (const g of grid) { const w = g.f * g.p * (1 - g.p); Ia += w; Ib += w * g.x * g.x; Iab += w * g.x; }
  const crit = [];
  crit.push({ label: 'Observados/esperados (O/E)', n: ceil((1 - phi) / (phi * seOE ** 2)), note: `SE de ln(O/E) ≤ ${seOE}` });
  crit.push({ label: 'Pendiente de calibración', n: ceil(Ia / (seSlope ** 2 * (Ia * Ib - Iab * Iab))), note: `SE ≤ ${seSlope}` });
  let N = 50; while (seC(C, phi, N) > seCstat && N < 1e8) N = N < 5000 ? N + 1 : ceil(N * 1.001);
  crit.push({ label: 'Estadístico C', n: N, note: `SE ≤ ${seCstat}` });
  const nb = [];
  for (const pt of thresholds) {
    const th = ln(pt / (1 - pt)); let tp = 0, fp = 0, tn = 0, fn = 0;
    for (const g of grid) { const e = g.f * g.p, ne = g.f * (1 - g.p); if (g.x >= th) { tp += e; fp += ne; } else { fn += e; tn += ne; } }
    const se = tp / (tp + fn), sp = tn / (tn + fp), w = ((1 - phi) / phi) * pt / (1 - pt), snb = se - w * (1 - sp), target = seSNB ?? snb / Z;
    const n = ceil((se * (1 - se) / phi + w * w * sp * (1 - sp) / (1 - phi) + w * w * (1 - sp) ** 2 / (phi * (1 - phi))) / target ** 2);
    nb.push({ pt, se, sp, snb, target, n });
  }
  if (nb.length) crit.push({ label: 'Beneficio neto estandarizado (umbrales: ' + nb.map(x => x.pt).join(', ') + ')', n: Math.max(...nb.map(x => x.n)), note: 'IC 95 % que excluya 0', detail: nb });
  const n = Math.max(...crit.map(c => c.n));
  return { crit, n, events: n * phi, phi, C, info: { Ia, Ib, Iab } };
}
function cFromGrid(grid, phi) { // C implícito de una distribución de LP (mezcla bien calibrada), por integración de la ROC
  let tp = 0, fp = 0, aucv = 0, E = 0, NE = 0;
  for (const g of grid) { E += g.f * g.p; NE += g.f * (1 - g.p); }
  let cumNE = 0;
  for (const g of grid) { const e = g.f * g.p / E, ne = g.f * (1 - g.p) / NE; aucv += e * (cumNE + 0.5 * ne); cumNE += ne; }
  return aucv;
}
export function valContinuous({ r2, varY, seR2 = 0.0255, seCITL = 1.25, seSlope = 0.051 }) {
  const crit = [
    { label: 'R²', n: ceil(4 * r2 * (1 - r2) ** 2 / seR2 ** 2), note: `SE ≤ ${seR2}` },
    { label: 'Calibración en general (CITL)', n: ceil(varY * (1 - r2) / seCITL ** 2), note: `SE ≤ ${seCITL}` },
    { label: 'Pendiente de calibración', n: ceil((1 - r2) / (seSlope ** 2 * r2) + 1), note: `SE ≤ ${seSlope}` },
    { label: 'Varianza residual (margen ≤ 10 %)', n: 235, note: 'mínimo de Archer et al.' },
  ];
  return { crit, n: Math.max(...crit.map(c => c.n)) };
}
// SE objetivo para un ancho de IC deseado
export const seFromWidth = w => w / (2 * Z);
