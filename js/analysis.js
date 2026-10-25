// Análisis descriptivo para el manuscrito: columna de agrupación por desenlace (Tabla 1) y regresión por predictor, cruda y ajustada.
// Mismo código en EPI-PREDIC y EPI-Maching-Learning.
import { fitGLM, fitCox, fitOLS, waldTerm } from './models.js?v=2026.10.25-5';
import { levelsOf, isMiss } from './data.js?v=2026.10.25-5';
import { qnorm } from './dist.js?v=2026.10.25-5';

// ---- columna de agrupación para la Tabla 1: binario → el desenlace; supervivencia → el estado; continuo → terciles
// spec: {type:'binary'|'survival'|'continuous', y, status}
export function groupColumn(ds, meta, spec) {
  if (spec.type === 'binary') return { ds, meta, group: spec.y, label: spec.y };
  if (spec.type === 'survival') return { ds, meta, group: spec.status, label: spec.status };
  const col = ds.cols[spec.y], v = Array.from(col).filter(isFinite).sort((a, b) => a - b), q = p => v[Math.floor((v.length - 1) * p)], c1 = q(1 / 3), c2 = q(2 / 3), nm = `${spec.y} (terciles)`, lv = ['T1 (bajo)', 'T2', 'T3 (alto)'];
  const g = Array.from(col, x => (isMiss(x) ? null : x <= c1 ? lv[0] : x <= c2 ? lv[1] : lv[2]));
  return { ds: { n: ds.n, cols: { ...ds.cols, [nm]: g } }, meta: { ...meta, [nm]: { name: nm, type: 'cat', levels: lv, ref: lv[0] } }, group: nm, label: nm };
}

// ---- regresión cruda (un predictor cada vez) y ajustada (todos a la vez), con los mismos pacientes (casos completos)
// opts: {type, vars:[nombres], y, event, time, status, sevent}; las cuantitativas entran de forma lineal (por unidad) y las categóricas como indicadores
export function crudeAdjusted(ds, meta, o) {
  const vars = o.vars.filter(v => meta[v] && (meta[v].type === 'num' || meta[v].type === 'cat')), need = [...vars, ...(o.type === 'survival' ? [o.time, o.status] : [o.y])], rows = [];
  for (let i = 0; i < ds.n; i++) if (need.every(v => !isMiss(ds.cols[v][i]))) rows.push(i);
  const colsOf = v => { const m = meta[v], col = ds.cols[v]; return m.type === 'num' ? [{ name: v, level: null, x: rows.map(i => +col[i]) }] : levelsOf(m).slice(1).map(l => ({ name: `${v}: ${l}`, level: l, x: rows.map(i => (String(col[i]) === l ? 1 : 0)) })); };
  const D = Object.fromEntries(vars.map(v => [v, colsOf(v)]));
  const yv = o.type === 'binary' ? Float64Array.from(rows, i => (String(ds.cols[o.y][i]) === String(o.event) ? 1 : 0)) : o.type === 'continuous' ? Float64Array.from(rows, i => +ds.cols[o.y][i]) : null;
  const tv = o.type === 'survival' ? Float64Array.from(rows, i => +ds.cols[o.time][i]) : null, sv = o.type === 'survival' ? Float64Array.from(rows, i => (String(ds.cols[o.status][i]) === String(o.sevent) ? 1 : 0)) : null;
  const events = o.type === 'binary' ? yv.reduce((a, b) => a + b, 0) : o.type === 'survival' ? sv.reduce((a, b) => a + b, 0) : null, z = qnorm(0.975);
  const fit = cols => { // devuelve {est[], lo[], hi[], p[], global}
    const n = rows.length, X = Array.from({ length: n }, (_, r) => Float64Array.from(cols.map(c => c.x[r])));
    try {
      const off = o.type === 'survival' ? 0 : 1, Xi = off ? X.map(x => Float64Array.of(1, ...x)) : X, f = o.type === 'binary' ? fitGLM(Xi, yv, 'binomial') : o.type === 'continuous' ? fitOLS(Xi, yv) : fitCox(Xi, tv, sv);
      return { f, off, ok: true };
    } catch { return { ok: false }; }
  };
  const eff = (F, j) => { const b = F.f.beta[j + F.off], se = F.f.se[j + F.off], t = o.type === 'continuous' ? v => v : Math.exp; return { est: t(b), lo: t(b - z * se), hi: t(b + z * se), p: F.f.p[j + F.off] }; };
  const all = vars.flatMap(v => D[v]), adj = fit(all), out = [];
  let pos = 0; const posOf = {}; vars.forEach(v => { posOf[v] = pos; pos += D[v].length; });
  for (const v of vars) {
    const cr = fit(D[v]), k = D[v].length, gl = F => (F.ok && k > 1 ? waldTerm(F.f, D[v].map((_, j) => j + F.off)).p : NaN);
    const globalCrude = gl(cr), globalAdj = adj.ok && k > 1 ? waldTerm(adj.f, D[v].map((_, j) => posOf[v] + j + adj.off)).p : NaN;
    if (k > 1) out.push({ name: v, header: true, pCrude: globalCrude, pAdj: globalAdj });
    D[v].forEach((c, j) => out.push({ name: k > 1 ? c.level : (c.level != null ? c.name : v), variable: v, level: c.level, ind: k > 1, crude: cr.ok ? eff(cr, j) : null, adj: adj.ok ? eff(adj, posOf[v] + j) : null }));
  }
  return { rows: out, n: rows.length, events, measure: o.type === 'binary' ? 'OR' : o.type === 'survival' ? 'HR' : 'β', adjOk: adj.ok, dropped: ds.n - rows.length };
}
const fmt = (v, d = 2) => (isFinite(v) ? (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(d)) : '–'), fpv = p => (!isFinite(p) ? '' : p < 0.001 ? '<0.001' : p.toFixed(3));
export const effTxt = (e, d = 2) => (e ? `${fmt(e.est, d)} (${fmt(e.lo, d)}–${fmt(e.hi, d)})` : '–');
// tabla lista para mostrar: [variable, crudo (IC), p, ajustado (IC), p]
export function uniAdjTable(R, lang = 'es') {
  const m = R.measure, es = lang === 'es', head = [es ? 'Variable' : 'Variable', es ? `${m} crudo (IC 95 %)` : `Crude ${m} (95% CI)`, 'p', es ? `${m} ajustado (IC 95 %)` : `Adjusted ${m} (95% CI)`, 'p'];
  return { head, rows: R.rows.map(r => (r.header ? [r.name, '', fpv(r.pCrude), '', fpv(r.pAdj)] : [(r.ind ? '   ' : '') + r.name, effTxt(r.crude), fpv(r.crude?.p), effTxt(r.adj), fpv(r.adj?.p)])), bold: R.rows.map(r => !!r.header) };
}
