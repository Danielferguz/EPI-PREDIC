// Pruebas de hipótesis bivariadas y de normalidad (Tabla 1 y supuestos).
import { qnorm, pnorm, pchisqU, pfU, p2t, lgamma, p2z } from './dist.js?v=2026.10.26-1';

export const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
export const variance = a => { const m = mean(a); return a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1); };
export const sd = a => Math.sqrt(variance(a));
export function quantile(a, q) {
  const s = Float64Array.from(a).sort(), h = (s.length - 1) * q, lo = Math.floor(h);
  return s.length ? s[lo] + (h - lo) * ((s[lo + 1] ?? s[lo]) - s[lo]) : NaN;
}
export const median = a => quantile(a, 0.5);

const poly = (c, x) => c.reduceRight((acc, v) => acc * x + v, 0);
// Shapiro-Wilk (Royston 1995, algoritmo AS R94, como shapiro.test de R)
export function shapiro(x) {
  x = Float64Array.from(x).sort(); const n = x.length;
  if (n < 3 || n > 5000) return { W: NaN, p: NaN };
  const range = x[n - 1] - x[0]; if (range < 1e-12) return { W: NaN, p: NaN };
  const nn2 = Math.floor(n / 2), a = new Float64Array(nn2 + 1);
  if (n === 3) a[1] = Math.SQRT1_2;
  else {
    const an25 = n + 0.25; let summ2 = 0;
    for (let i = 1; i <= nn2; i++) { a[i] = qnorm((i - 0.375) / an25); summ2 += a[i] * a[i]; }
    summ2 *= 2; const ssumm2 = Math.sqrt(summ2), rsn = 1 / Math.sqrt(n);
    const a1 = poly([0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056], rsn) - a[1] / ssumm2;
    let i1, fac;
    if (n > 5) {
      i1 = 3; const a2 = -a[2] / ssumm2 + poly([0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633], rsn);
      fac = Math.sqrt((summ2 - 2 * a[1] ** 2 - 2 * a[2] ** 2) / (1 - 2 * a1 ** 2 - 2 * a2 ** 2)); a[2] = a2;
    } else { i1 = 2; fac = Math.sqrt((summ2 - 2 * a[1] ** 2) / (1 - 2 * a1 ** 2)); }
    a[1] = a1; for (let i = i1; i <= nn2; i++) a[i] /= -fac;
  }
  const coef = new Float64Array(n);
  for (let i = 1; i <= nn2; i++) { coef[i - 1] = -a[i]; coef[n - i] = a[i]; }
  const xm = mean(x), cm = mean(coef); let sxx = 0, saa = 0, sax = 0;
  for (let i = 0; i < n; i++) { const dx = x[i] - xm, da = coef[i] - cm; sxx += dx * dx; saa += da * da; sax += da * dx; }
  const W = sax * sax / (saa * sxx), w1 = 1 - W;
  if (n === 3) return { W, p: Math.max(0, 1.90985931710274 * (Math.asin(Math.sqrt(W)) - 1.04719755119660)) };
  let y = Math.log(w1), m, s;
  if (n <= 11) {
    const g = poly([-2.273, 0.459], n); if (y >= g) return { W, p: 1e-99 };
    y = -Math.log(g - y); m = poly([0.544, -0.39978, 0.025054, -6.714e-4], n); s = Math.exp(poly([1.3822, -0.77857, 0.062767, -0.0020322], n));
  } else { const xx = Math.log(n); m = poly([-1.5861, -0.31082, -0.083751, 0.0038915], xx); s = Math.exp(poly([-0.4803, -0.082676, 0.0030302], xx)); }
  return { W, p: 1 - pnorm((y - m) / s) };
}

// chi² de Pearson para tabla r×c (sin corrección de continuidad)
export function chisqTest(tab) {
  const R = tab.length, C = tab[0].length, rs = tab.map(r => r.reduce((a, b) => a + b, 0)), cs = tab[0].map((_, j) => tab.reduce((a, r) => a + r[j], 0)), N = rs.reduce((a, b) => a + b, 0);
  let x2 = 0, minE = Infinity, small = 0;
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) { const e = rs[i] * cs[j] / N; if (e === 0) continue; minE = Math.min(minE, e); if (e < 5) small++; x2 += (tab[i][j] - e) ** 2 / e; }
  const df = (rs.filter(v => v > 0).length - 1) * (cs.filter(v => v > 0).length - 1);
  return { stat: x2, df, p: df > 0 ? pchisqU(x2, df) : NaN, minE, propSmall: small / (R * C) };
}
// Fisher exacto bilateral para 2×2
export function fisher2x2([[a, b], [c, d]]) {
  const r1 = a + b, r2 = c + d, c1 = a + c, N = r1 + r2;
  const lf = k => lgamma(k + 1), lp = x => lf(r1) + lf(r2) + lf(c1) + lf(N - c1) - lf(N) - lf(x) - lf(r1 - x) - lf(c1 - x) - lf(r2 - c1 + x);
  const p0 = lp(a); let p = 0;
  for (let x = Math.max(0, c1 - r2); x <= Math.min(r1, c1); x++) { const lx = lp(x); if (lx <= p0 + 1e-7) p += Math.exp(lx); }
  return { p: Math.min(1, p) };
}
export function welch(x, y) {
  const m1 = mean(x), m2 = mean(y), v1 = variance(x) / x.length, v2 = variance(y) / y.length, t = (m1 - m2) / Math.sqrt(v1 + v2);
  const df = (v1 + v2) ** 2 / (v1 * v1 / (x.length - 1) + v2 * v2 / (y.length - 1));
  return { stat: t, df, p: p2t(t, df) };
}
export function anova(groups) {
  const all = groups.flat(), gm = mean(all), k = groups.length, N = all.length;
  let ssb = 0, ssw = 0; groups.forEach(g => { const m = mean(g); ssb += g.length * (m - gm) ** 2; g.forEach(v => (ssw += (v - m) ** 2)); });
  const F = (ssb / (k - 1)) / (ssw / (N - k));
  return { stat: F, df: [k - 1, N - k], p: pfU(F, k - 1, N - k) };
}
function ranks(all) {
  const idx = all.map((v, i) => i).sort((i, j) => all[i] - all[j]), r = new Float64Array(all.length); let ties = 0;
  for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && all[idx[j + 1]] === all[idx[i]]) j++; const rr = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[idx[k]] = rr; const t = j - i + 1; ties += t ** 3 - t; i = j + 1; }
  return { r, ties };
}
// Kruskal-Wallis con corrección por empates
export function kruskal(groups) {
  const all = groups.flat(), N = all.length, { r, ties } = ranks(all); let off = 0, H = 0;
  groups.forEach(g => { let s = 0; for (let i = 0; i < g.length; i++) s += r[off + i]; off += g.length; H += s * s / g.length; });
  H = (12 / (N * (N + 1)) * H - 3 * (N + 1)) / (1 - ties / (N ** 3 - N));
  return { stat: H, df: groups.length - 1, p: pchisqU(H, groups.length - 1) };
}
// Mann-Whitney: aproximación normal con corrección por empates y de continuidad (como wilcox.test de R)
export function mannWhitney(x, y) {
  const all = [...x, ...y], N = all.length, { r, ties } = ranks(all); let R1 = 0; for (let i = 0; i < x.length; i++) R1 += r[i];
  const U = R1 - x.length * (x.length + 1) / 2, mu = x.length * y.length / 2;
  const s = Math.sqrt(x.length * y.length / 12 * ((N + 1) - ties / (N * (N - 1))));
  const z = (U - mu - 0.5 * Math.sign(U - mu)) / s;
  return { stat: U, p: p2z(z) };
}
