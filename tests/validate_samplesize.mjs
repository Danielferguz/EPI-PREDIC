// Validación de js/samplesize.js: (1) frente al puerto de Python de pmsampsize (tests/reference_samplesize.json),
// (2) frente a los ejemplos publicados (Riley, diapositivas; BMJ 2024;384:e074821) y (3) por simulación de Monte Carlo (validación externa).
import { readFileSync } from 'node:fs';
import { devBinary, devSurvival, devContinuous, cstatToR2cs, valBinary, valContinuous, seC, lpDistribution, seFromWidth } from '../js/samplesize.js';
import { rng } from '../js/dist.js';
import { fitGLM } from '../js/models.js';

const ref = JSON.parse(readFileSync(new URL('reference_samplesize.json', import.meta.url)));
let fails = 0, checks = 0;
const ok = (name, cond, extra = '') => { checks++; if (!cond) { fails++; console.log('FAIL', name, extra); } else console.log('ok  ', name); };

// 1. pmsampsize (Python)
let bad = 0;
for (const r of ref.binary) { const o = devBinary({ r2cs: r.r2cs, p: r.p, prev: r.prev }); if (o.n !== r.n || o.crit.some((c, i) => c.n !== r.crit[i])) { bad++; console.log('  bin', JSON.stringify(r), o.crit.map(c => c.n), o.n); } }
ok(`binario: ${ref.binary.length} escenarios idénticos a pmsampsize`, bad === 0, `${bad} difieren`);
bad = 0; for (const r of ref.survival) { const o = devSurvival({ r2cs: r.r2cs, p: r.p, rate: r.rate, timepoint: r.timepoint, meanfup: r.meanfup }); if (o.crit[0].n !== r.crit[0] || o.crit[1].n !== r.crit[1] || o.n !== r.n) { bad++; console.log('  surv', JSON.stringify(r), o.crit.map(c => c.n), o.n); } }
ok(`supervivencia: ${ref.survival.length} escenarios (criterios 1-2 y n final)`, bad === 0, `${bad} difieren`);
bad = 0; for (const r of ref.continuous) { const o = devContinuous({ r2: r.r2, p: r.p, intercept: 50, sd: 5 }); if (o.crit.slice(0, 3).some((c, i) => c.n !== r.crit[i]) || o.n !== r.n) { bad++; console.log('  cont', JSON.stringify(r), o.crit.map(c => c.n), o.n); } }
ok(`continuo: ${ref.continuous.length} escenarios`, bad === 0, `${bad} difieren`);
for (const r of ref.cstat) { const v = cstatToR2cs(r.c, r.prev_events); ok(`R² de Cox-Snell desde C=${r.c}, prevalencia ${r.prev_events}`, Math.abs(v - r.r2cs) < 0.004, `${v.toFixed(4)} vs ${r.r2cs.toFixed(4)} (simulación de Python)`); }

// 2. ejemplos publicados
const e1 = devBinary({ r2cs: 0.2, p: 30, prev: 0.5 }); ok('Riley (diapositiva 19): 1194 / 701 / 385', e1.crit.map(c => c.n).join() === '1194,701,385' && e1.n === 1194 && Math.abs(e1.epp - 19.9) < 0.05, e1.crit.map(c => c.n));
const e2 = devBinary({ r2cs: 0.5, p: 30, prev: 0.5 }); ok('Riley (diapositiva 21): 370 / ≈556 / 385 (558 con la versión actual de pmsampsize)', e2.crit[0].n === 370 && Math.abs(e2.crit[1].n - 556) <= 3 && e2.crit[2].n === 385 && e2.n === e2.crit[1].n, e2.crit.map(c => c.n));
const e3 = devSurvival({ r2cs: 0.051, p: 30, rate: 0.065, timepoint: 2, meanfup: 2.07 }); ok('pmsampsize (ejemplo de supervivencia): 5143 sujetos, EPP 23.07', e3.n === 5143 && Math.abs(e3.epp - 23.07) < 0.01);
const v1 = valContinuous({ r2: 0.4, varY: 497.29, seR2: 0.0255, seCITL: seFromWidth(5), seSlope: seFromWidth(0.3) });
ok('BMJ 2024 (parte 3): 886 / 184 / 258 / 235', v1.crit.map(c => c.n).join() === '886,184,258,235', v1.crit.map(c => c.n));

// 3. Monte Carlo de la validación binaria: con N calculado, el error estándar empírico debe coincidir con el objetivo
function sim(prev, cstat, N, B, seed = 7) {
  const { grid } = lpDistribution({ prev, cstat }), r = rng(seed), cdf = []; let acc = 0; for (const g of grid) { acc += g.f; cdf.push(acc); }
  const slopes = [], oes = [], cs = [];
  for (let b = 0; b < B; b++) {
    const x = new Float64Array(N), y = new Float64Array(N);
    for (let i = 0; i < N; i++) { const u = r.u() * acc; let lo = 0, hi = cdf.length - 1; while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; } x[i] = grid[lo].x; y[i] = r.u() < 1 / (1 + Math.exp(-x[i])) ? 1 : 0; }
    const X = Array.from(x, v => Float64Array.of(1, v)); slopes.push(fitGLM(X, y, 'binomial').beta[1]);
    let O = 0, E = 0; for (let i = 0; i < N; i++) { O += y[i]; E += 1 / (1 + Math.exp(-x[i])); } oes.push(Math.log(O / E));
    // C de Mann-Whitney
    const idx = Array.from(x.keys()).sort((a, c) => x[a] - x[c]); let rsum = 0, n1 = 0; idx.forEach((id, k) => { if (y[id]) { rsum += k + 1; n1++; } }); cs.push((rsum - n1 * (n1 + 1) / 2) / (n1 * (N - n1)));
  }
  const sd = a => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); };
  return { slope: sd(slopes), oe: sd(oes), c: sd(cs), cMean: cs.reduce((s, v) => s + v, 0) / B };
}
const target = { seOE: 0.1, seSlope: 0.08, seCstat: 0.02 }, v = valBinary({ prev: 0.2, cstat: 0.8, ...target });
const mc = [['pendiente', 'slope', v.crit[1].n, target.seSlope], ['ln(O/E)', 'oe', v.crit[0].n, target.seOE], ['C', 'c', v.crit[2].n, target.seCstat]].map(([nm, k, N, se]) => ({ nm, k, N, se, emp: sim(0.2, 0.8, N, 600)[k] }));
for (const m of mc) ok(`Monte Carlo ${m.nm}: N=${m.N}, SE empírico ${m.emp.toFixed(4)} vs objetivo ${m.se}`, Math.abs(m.emp / m.se - 1) < 0.12, m.emp);
ok('C implícito del LP normal ≈ C solicitado', Math.abs(valBinary({ prev: 0.2, cstat: 0.8 }).C - 0.8) < 1e-9);
const vn = valBinary({ prev: NaN, cstat: NaN, lpMean: -1.75, lpSd: 1.47 }); ok('LP ~ N(−1.75, 1.47²): prevalencia implícita ≈ 0.22 y C razonable', Math.abs(vn.phi - 0.22) < 0.03 && vn.C > 0.7 && vn.C < 0.9, `${vn.phi.toFixed(3)} / ${vn.C.toFixed(3)}`);
console.log(`\n${checks - fails}/${checks} comprobaciones correctas`); process.exit(fails ? 1 : 0);
