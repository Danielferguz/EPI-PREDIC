// Simplificación del modelo: predictores ordenados por su aporte (χ² de Wald parcial − gl del modelo completo) y rendimiento corregido por bootstrap
// al añadirlos de uno en uno. En cada réplica se repite el orden y el ajuste (sin optimismo por selección). Se propone el modelo más pequeño cuyo
// rendimiento corregido no baja más que una tolerancia respecto al completo.
import { fitPred, fitFixedLambda, evaluate, tick } from './predmodel.js?v=2026.10.25-8';
import { waldTerm } from './models.js?v=2026.10.25-8';
import { rng } from './dist.js?v=2026.10.25-8';

const KEY = { binary: 'auc', survival: 'cindex', continuous: 'r2' };
// orden de importancia: refit sin penalizar solo para ordenar (la penalización no da errores estándar)
export function rankVars(ds, meta, spec, rows, seed = 1) {
  const sp = { ...spec, penalty: 'none' };
  try {
    const m = fitPred(ds, meta, sp, rows, { seed, quiet: true }), off = m.type === 'survival' ? 0 : 1, out = [];
    for (const v of spec.vars) { const idx = m.cols.map((c, j) => (c.v === v.name ? j + off : -1)).filter(j => j >= 0), w = waldTerm(m.fit, idx), chi = w.F ? w.stat * w.df : w.stat; out.push({ name: v.name, chi2: chi, df: idx.length, score: chi - idx.length }); }
    return out.sort((a, b) => b.score - a.score);
  } catch { return spec.vars.map(v => ({ name: v.name, chi2: NaN, df: 1, score: 0 })); }
}
const subSpec = (spec, names) => ({ ...spec, vars: names.map(n => spec.vars.find(v => v.name === n)) });
function fitK(ds, meta, spec, rows, lambda, seed) {
  return lambda != null && spec.penalty && spec.penalty !== 'none' ? fitFixedLambda(ds, meta, spec, rows, lambda) : fitPred(ds, meta, spec, rows, { seed, quiet: true });
}
// ds, meta, spec: ya materializados (state.dsX, state.metaX, state.specX); model: modelo completo ajustado
export async function simplifyPath(ds, meta, spec, rows, model, { B = 100, seed = 7, tol = 0.01, onProgress = null } = {}) {
  const key = KEY[spec.type], R = rng(seed), horizon = spec.horizon, lam = model.pen?.lambda ?? null, p = spec.vars.length, n = rows.length;
  const imp = rankVars(ds, meta, spec, rows, seed), order = imp.map(i => i.name), app = [];
  for (let k = 1; k <= p; k++) { const sp = subSpec(spec, order.slice(0, k)), m = fitK(ds, meta, sp, rows, lam, seed); app.push(evaluate(m, ds, meta, rows, { horizon }).perf[key]); }
  const opt = Array.from({ length: p }, () => []); let failed = 0;
  for (let b = 0; b < B; b++) {
    const samp = Array.from({ length: n }, () => rows[R.int(n)]);
    try {
      const ob = rankVars(ds, meta, spec, samp, seed + b).map(i => i.name), o = [];
      for (let k = 1; k <= p; k++) { const sp = subSpec(spec, ob.slice(0, k)), m = fitK(ds, meta, sp, samp, lam, seed + b), a = evaluate(m, ds, meta, samp, { horizon }).perf[key], t = evaluate(m, ds, meta, rows, { horizon }).perf[key]; if (!isFinite(a) || !isFinite(t)) throw new Error('no finita'); o.push(a - t); }
      o.forEach((v, k) => opt[k].push(v));
    } catch { failed++; }
    if (b % 4 === 3) { onProgress?.((b + 1) / B); await tick(); }
  }
  const mean = a => a.reduce((s, v) => s + v, 0) / (a.length || 1), tab = order.map((name, i) => ({ k: i + 1, added: name, vars: order.slice(0, i + 1), apparent: app[i], optimism: mean(opt[i]), corrected: app[i] - mean(opt[i]) }));
  const full = tab[p - 1].corrected, kStar = (tab.find(r => r.corrected >= full - tol) || tab[p - 1]).k;
  return { key, imp, order, rows: tab, kStar, tol, B: B - failed, failed, full };
}
