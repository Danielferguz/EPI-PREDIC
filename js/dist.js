// Funciones de distribución (normal, t, chi², F), gamma y beta incompletas, generador aleatorio con semilla.

const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1; let a = LANCZOS[0]; const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// gamma incompleta regularizada P(a, x) y Q(a, x)
export function gammaP(a, x) {
  if (x <= 0) return 0;
  if (x < a + 1) {
    let sum = 1 / a, del = sum, ap = a;
    for (let n = 0; n < 1000; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-16) break; }
    return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
  }
  return 1 - gammaQcf(a, x);
}
export function gammaQ(a, x) { return x <= 0 ? 1 : x < a + 1 ? 1 - gammaP(a, x) : gammaQcf(a, x); }
function gammaQcf(a, x) {
  const FP = 1e-300; let b = x + 1 - a, c = 1 / FP, d = 1 / b, h = d;
  for (let i = 1; i < 1000; i++) {
    const an = -i * (i - a); b += 2;
    d = an * d + b; if (Math.abs(d) < FP) d = FP;
    c = b + an / c; if (Math.abs(c) < FP) c = FP;
    d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-16) break;
  }
  return Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
}

// beta incompleta regularizada I_x(a, b)
export function betaI(x, a, b) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(x, a, b) / a : 1 - bt * betacf(1 - x, b, a) / b;
}
function betacf(x, a, b) {
  const FP = 1e-300, qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FP) d = FP; d = 1 / d; let h = d;
  for (let m = 1; m <= 3000; m++) {
    const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d;
    const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-16) break;
  }
  return h;
}

export function erfc(x) { return x >= 0 ? gammaQ(0.5, x * x) : 2 - gammaQ(0.5, x * x); }
export const pnorm = z => 0.5 * erfc(-z / Math.SQRT2);
export function qnorm(p) {
  if (p <= 0) return -Infinity; if (p >= 1) return Infinity;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  let x;
  if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p)); x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  else if (p > 1 - 0.02425) { const q = Math.sqrt(-2 * Math.log(1 - p)); x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  else { const q = p - 0.5, r = q * q; x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1); }
  // un paso de Halley
  const e = pnorm(x) - p, u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
  return x - u / (1 + x * u / 2);
}
export const dnorm = z => Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);

export function pt(t, df) {
  if (!isFinite(df) || df > 1e7) return pnorm(t);
  const x = df / (df + t * t), tail = 0.5 * betaI(x, df / 2, 0.5);
  return t > 0 ? 1 - tail : tail;
}
export function qt(p, df) {
  if (!isFinite(df) || df > 1e7) return qnorm(p);
  let lo = -1e3, hi = 1e3, x = qnorm(p);
  for (let i = 0; i < 200; i++) {
    const f = pt(x, df) - p;
    if (Math.abs(f) < 1e-14) break;
    if (f > 0) hi = x; else lo = x;
    const dens = Math.exp(lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * Math.log(df * Math.PI) - (df + 1) / 2 * Math.log(1 + x * x / df));
    let nx = x - f / dens; if (!(nx > lo && nx < hi)) nx = (lo + hi) / 2; x = nx;
  }
  return x;
}
export const pchisq = (x, df) => (x <= 0 ? 0 : gammaP(df / 2, x / 2));
export const pchisqU = (x, df) => (x <= 0 ? 1 : gammaQ(df / 2, x / 2));
export function qchisq(p, df) {
  let lo = 0, hi = Math.max(10, df * 10);
  while (pchisq(hi, df) < p) hi *= 2;
  for (let i = 0; i < 200; i++) { const m = (lo + hi) / 2; if (pchisq(m, df) < p) lo = m; else hi = m; if (hi - lo < 1e-12 * hi) break; }
  return (lo + hi) / 2;
}
export const pfU = (f, d1, d2) => (f <= 0 ? 1 : betaI(d2 / (d2 + d1 * f), d2 / 2, d1 / 2));
export const p2t = (t, df) => 2 * (1 - pt(Math.abs(t), df));
export const p2z = z => erfc(Math.abs(z) / Math.SQRT2);

// ---------- aleatorios con semilla (mulberry32) ----------
export function rng(seed = 2024) {
  let a = seed >>> 0, spare = null;
  const u = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const norm = () => {
    if (spare != null) { const s = spare; spare = null; return s; }
    let x, y, s; do { x = 2 * u() - 1; y = 2 * u() - 1; s = x * x + y * y; } while (s >= 1 || s === 0);
    const m = Math.sqrt(-2 * Math.log(s) / s); spare = y * m; return x * m;
  };
  const gamma = k => { // Marsaglia-Tsang
    if (k < 1) return gamma(k + 1) * Math.pow(u(), 1 / k);
    const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) { let x, v; do { x = norm(); v = 1 + c * x; } while (v <= 0); v = v * v * v; const uu = u(); if (uu < 1 - 0.0331 * x ** 4 || Math.log(uu) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; }
  };
  return { u, norm, chisq: df => 2 * gamma(df / 2), int: n => Math.floor(u() * n) };
}
