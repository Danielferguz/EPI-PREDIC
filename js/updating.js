// Actualización de un modelo en datos nuevos (Binuya et al. 2022): recalibración, revisión y extensión, comparadas con validación cruzada
// dentro de la base de validación (así el rendimiento de cada estrategia no es el aparente de los mismos datos con los que se actualizó).
import { recalibrate, evaluate, fitPred, perfBinary, perfContinuous } from './predmodel.js?v=2026.10.26-1';
import { isMiss } from './data.js?v=2026.10.26-1';
import { rng } from './dist.js?v=2026.10.26-1';

export const STRATEGIES = [
  ['orig', 'Modelo original (sin actualizar)', 0], ['citl', 'Recalibración en general (solo intercepto)', 1], ['slope', 'Recalibración logística (intercepto y pendiente)', 2],
  ['rev', 'Revisión: reestimar todos los coeficientes', null], ['ext', 'Extensión: revisión con un marcador nuevo', null]];

// modelo actualizado con las filas `rows` de (ds, meta); `extra` = nombre de la variable que se añade en la extensión
export function applyUpdate(id, m, ds, meta, rows, extra = null) {
  if (id === 'orig') return m;
  if (id === 'citl' || id === 'slope') return recalibrate(m, ds, meta, rows, id);
  const spec = id === 'rev' ? m.spec : { ...m.spec, vars: [...m.spec.vars, { name: extra, form: 'lin' }] };
  return fitPred(ds, meta, spec, rows, { quiet: true, seed: 1, folds: 5 });
}
const nParams = (id, m) => ({ orig: 0, citl: 1, slope: 2 }[id] ?? (m ? m.cols.length + (m.type === 'survival' ? 0 : 1) : NaN));
// variables que se pueden añadir: están en los datos de desarrollo y de validación, no son el desenlace y aún no están en el modelo
export function extensionCandidates(m, meta, ds) {
  const used = new Set([...m.spec.vars.map(v => v.name), m.spec.y, m.spec.time, m.spec.status].filter(Boolean));
  return Object.keys(meta).filter(k => meta[k].type !== 'skip' && !used.has(k) && ds.cols[k]);
}
// pliegues (estratificados por evento si es binario)
function folds(n, K, y, seed) {
  const R = rng(seed), idx = Array.from({ length: n }, (_, i) => i), f = new Int32Array(n), groups = y ? [idx.filter(i => y[i] === 1), idx.filter(i => y[i] !== 1)] : [idx];
  for (const g of groups) { g.sort(() => R.u() - 0.5); g.forEach((i, k) => (f[i] = k % K)); }
  return f;
}
// rendimiento por validación cruzada repetida: para cada estrategia se actualiza con K−1 pliegues y se evalúa en el restante; las predicciones se juntan
export function cvCompare({ ids, m, ds, meta, rows, extra = null, K = 5, reps = 3, seed = 7 }) {
  const type = m.type, out = {};
  let use = rows; if (ids.includes('ext') && extra) use = rows.filter(i => !isMiss(ds.cols[extra][i]));
  if (use.length < 40) throw new Error('Muy pocas filas completas para comparar estrategias.');
  const yAll = type === 'binary' ? Float64Array.from(use, i => (String(ds.cols[m.spec.y][i]) === String(m.spec.event) ? 1 : 0)) : null;
  for (const id of ids) {
    const acc = []; let err = null;
    for (let r = 0; r < reps && !err; r++) {
      const f = folds(use.length, K, yAll, seed + 101 * r), P = [], Y = [];
      try {
        for (let k = 0; k < K; k++) {
          const tr = use.filter((_, j) => f[j] !== k), te = use.filter((_, j) => f[j] === k), mk = applyUpdate(id, m, ds, meta, tr, extra), ev = evaluate(mk, ds, meta, te);
          ev.p.forEach((v, j) => { P.push(v); Y.push(ev.y[j]); });
        }
        acc.push(type === 'binary' ? perfBinary(Float64Array.from(P), Float64Array.from(Y)) : perfContinuous(Float64Array.from(P), Float64Array.from(Y)));
      } catch (e) { err = e.message || String(e); }
    }
    const mean = key => acc.reduce((s, a) => s + a[key], 0) / (acc.length || 1);
    out[id] = err ? { err } : type === 'binary' ? { auc: mean('auc'), slope: mean('slope'), citl: mean('citl'), oe: mean('oe'), brier: mean('brier'), n: use.length } : { r2: mean('r2'), slope: mean('slope'), citl: mean('citl'), rmse: mean('rmse'), n: use.length };
  }
  return { res: out, n: use.length };
}
// la estrategia más sencilla con calibración aceptable (pendiente 0.9–1.1 y |CITL| < 0.1; en continuo, la CITL relativa a la DE); si ninguna, la de menor Brier/RMSE
export function recommend(type, res, order) {
  const ok = r => r && !r.err && r.slope >= 0.9 && r.slope <= 1.1 && Math.abs(r.citl) < (type === 'binary' ? 0.1 : 0.1 * (r.rmse || 1)), first = order.find(id => ok(res[id]));
  if (first) return { id: first, why: 'la más sencilla con pendiente entre 0.9 y 1.1 e intercepto cercano a 0' };
  const key = type === 'binary' ? 'brier' : 'rmse', cand = order.filter(id => res[id] && !res[id].err).sort((a, b) => res[a][key] - res[b][key]);
  return cand.length ? { id: cand[0], why: `ninguna alcanza una calibración ideal; es la de menor ${type === 'binary' ? 'Brier' : 'RMSE'}` } : null;
}
