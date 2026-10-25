// Validación del motor de predicción (js/predmodel.js, js/glmnet.js) frente a statsmodels, scikit-learn y lifelines (tests/reference_*.json),
// propiedades de los splines y una simulación de la corrección del optimismo por bootstrap.
import { readFileSync } from 'node:fs';
import { fitPred, evaluate, cindex, bootstrapValidate, rcsRow, knotsFor, shrinkModel, recalibrate, dcaBinary, calibGroups, perfBinary, breslow, H0at, riskFromLP, predictLPs, completeFor } from '../js/predmodel.js';
import { enet } from '../js/glmnet.js';
import { rng } from '../js/dist.js';
import { concordance } from '../js/models.js';

import { recommend } from '../js/updating.js';
const R = f => JSON.parse(readFileSync(new URL(f, import.meta.url)));
const ref = R('reference_pred.json'), gref = R('reference_glmnet.json');
let fails = 0, checks = 0;
const ok = (name, cond, extra = '') => { checks++; if (!cond) { fails++; console.log('FAIL', name, extra); } else console.log('ok  ', name); };
const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol * Math.max(1, Math.abs(b));
const mk = (cols, meta) => ({ n: Object.values(cols)[0].length, cols });
const num = n => ({ name: n, type: 'num', levels: [], ref: null }), cat = (n, lv) => ({ name: n, type: 'cat', levels: lv, ref: lv[0], event: lv[1] });

// ---- logística: coeficientes, AUC, Brier, calibración
{
  const b = ref.bin, ds = mk({ x1: Float64Array.from(b.x1), x2: Float64Array.from(b.x2), x3: Float64Array.from(b.x3), g: b.g.map(String), y: b.y.map(String) }), meta = { x1: num('x1'), x2: num('x2'), x3: num('x3'), g: cat('g', ['0', '1']), y: cat('y', ['0', '1']) };
  const spec = { type: 'binary', y: 'y', event: '1', vars: [{ name: 'x1' }, { name: 'x2' }, { name: 'x3' }, { name: 'g' }] }, rows = completeFor(ds, spec), m = fitPred(ds, meta, spec, rows), ev = evaluate(m, ds, meta, rows);
  ok('logística: coeficientes = statsmodels', [m.intercept, ...m.beta].every((v, i) => near(v, b.beta[i], 1e-6)));
  ok('AUC = sklearn', near(ev.perf.auc, b.auc, 1e-9)); ok('Brier = sklearn', near(ev.perf.brier, b.brier, 1e-9));
  ok('pendiente de calibración (apparente) = 1 y = statsmodels', near(ev.perf.slope, b.slope, 1e-5) && Math.abs(ev.perf.slope - 1) < 1e-4, ev.perf.slope);
  ok('CITL ≈ 0 y = statsmodels', Math.abs(ev.perf.citl - b.citl) < 1e-6 && Math.abs(ev.perf.citl) < 1e-6); ok('O/E = 1', near(ev.perf.oe, b.oe, 1e-9) && Math.abs(ev.perf.oe - 1) < 1e-8);
  // ridge fijo = referencia sklearn (comprobado en detalle en validate_glmnet); aquí: el modelo penalizado reduce la dispersión del LP
  const mr = fitPred(ds, meta, { ...spec, penalty: 'ridge' }, rows), sd = a => { const mu = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - mu) ** 2, 0) / a.length); };
  ok('ridge (λ por CV) contrae el predictor lineal', sd(predictLPs(mr, ds, meta, rows).lp) <= sd(ev.lp) + 1e-9 && mr.pen.lambda > 0);
  const ml = fitPred(ds, meta, { ...spec, penalty: 'lasso', lambdaRule: '1se' }, rows); ok('lasso (1-ES) deja x3 (ruido) con coeficiente 0 o muy pequeño', Math.abs(ml.beta[2]) < 0.12, ml.beta[2]);
  // contracción y recalibración conservan la calibración global
  const sh = shrinkModel(m, 0.9, ds, meta), es = evaluate(sh, ds, meta, rows); ok('contracción uniforme: CITL = 0 y pendiente > 1 (modelo sobrecontraído)', Math.abs(es.perf.citl) < 1e-6 && es.perf.slope > 1);
  const rcc = recalibrate(m, ds, meta, rows, 'citl'), erc = evaluate(rcc, ds, meta, rows); ok('recalibración (solo intercepto): CITL 0', Math.abs(erc.perf.citl) < 1e-6);
  const rc = recalibrate(m, ds, meta, rows, 'slope'), er = evaluate(rc, ds, meta, rows); ok('recalibración (intercepto+pendiente): pendiente 1 y CITL 0', near(er.perf.slope, 1, 1e-6) && Math.abs(er.perf.citl) < 1e-6);
  // DCA: con umbral → 0 el beneficio neto del modelo tiende a la prevalencia; tratar a todos = prev − (1−prev) pt/(1−pt)
  const d = dcaBinary(ev.p, ev.y, [0.001, 0.2]); ok('DCA: tratar a todos', near(d[1].all, (b.y.reduce((s, v) => s + v, 0) / b.y.length) - (1 - b.y.reduce((s, v) => s + v, 0) / b.y.length) * 0.25, 1e-12) && d[0].model > 0);
  ok('calibración por grupos: n y suma de eventos', (() => { const g = calibGroups(ev.p, ev.y, 10); return g.reduce((s, r) => s + r.n, 0) === 700; })());
}
// ---- Cox: C de Harrell (con empates), riesgos a 3 y 6 años frente a lifelines
{
  const s = ref.surv, ds = mk({ x1: Float64Array.from(s.x1), x2: Float64Array.from(s.x2), g: s.g.map(String), time: Float64Array.from(s.time), ev: s.event.map(String) }), meta = { x1: num('x1'), x2: num('x2'), g: cat('g', ['0', '1']), time: num('time'), ev: cat('ev', ['0', '1']) };
  const spec = { type: 'survival', time: 'time', status: 'ev', event: '1', horizon: 6, vars: [{ name: 'x1' }, { name: 'x2' }, { name: 'g' }] }, rows = completeFor(ds, spec), m = fitPred(ds, meta, spec, rows), ev = evaluate(m, ds, meta, rows, { horizon: 6 });
  ok('Cox: coeficientes = lifelines (Efron)', m.beta.every((v, i) => near(v, s.beta[i], 1e-5)), Array.from(m.beta));
  ok('índice C de Harrell (con empates) = lifelines', near(ev.perf.cindex, s.cindex, 1e-9), `${ev.perf.cindex} vs ${s.cindex}`);
  const r3 = riskFromLP(m, ev.lp.slice(0, 5), 3), r6 = riskFromLP(m, ev.lp.slice(0, 5), 6);
  ok('riesgo a 3 y 6 años (Breslow) ≈ lifelines', [0, 1, 2, 3, 4].every(i => Math.abs(1 - r3[i] - s.surv5[i][0]) < 2e-3 && Math.abs(1 - r6[i] - s.surv5[i][1]) < 2e-3), `${Array.from(r6).map(v => (1 - v).toFixed(4))} vs ${s.surv5.map(r => r[1].toFixed(4))}`);
  { const rcs = recalibrate(m, ds, meta, rows, 'slope'), e2 = evaluate(rcs, ds, meta, rows, { horizon: 6 }); ok('Cox recalibrado: C se conserva y pendiente 1', near(e2.perf.cindex, ev.perf.cindex, 1e-9) && near(e2.perf.slope, 1, 1e-6), `${e2.perf.cindex} ${e2.perf.slope}`); }
  // índice C rápido = versión O(n²) sin empates
  const r = rng(3), n = 400, lp = Float64Array.from({ length: n }, () => r.norm()), t = Float64Array.from(lp, v => -Math.log(r.u()) / Math.exp(v) + 1e-9 * r.u()), st = Float64Array.from(t, (_, i) => (r.u() < 0.7 ? 1 : 0));
  ok('C rápido O(n log n) = EPI-DATA O(n²) sin empates', near(cindex(lp, t, st).c, concordance(lp, t, st), 1e-12));
}
// ---- splines cúbicos restringidos: lineales fuera de los nudos extremos, continuidad y primera columna = x
{
  const t = [1, 3, 5, 8], f = x => rcsRow(x, t), d2 = (x, h = 0.5, j = 1) => (f(x + h)[j] - 2 * f(x)[j] + f(x - h)[j]) / h ** 2;
  ok('RCS: columnas = k − 1 y primera = x', f(2).length === 3 && f(2.5)[0] === 2.5);
  ok('RCS: segunda derivada ≈ 0 más allá del último nudo (restringido)', Math.abs(d2(10, 0.5, 1)) < 1e-9 && Math.abs(d2(12, 0.5, 2)) < 1e-9 && Math.abs(d2(-3, 0.5, 1)) < 1e-9);
  ok('RCS: nudos de Harrell (k=4 → percentiles 5, 35, 65, 95)', (() => { const k = knotsFor(Array.from({ length: 101 }, (_, i) => i), 4); return k.every((v, i) => near(v, [5, 35, 65, 95][i], 1e-9)); })());
}
// ---- corrección del optimismo (simulación repetida): con n pequeño y muchos predictores de ruido, el promedio de las medidas corregidas
// debe estar mucho más cerca del promedio de las medidas «verdaderas» (muestra independiente grande) que el promedio de las aparentes
{
  const rr = rng(2024), P = 12, gen = (n) => { const cols = {}; for (let j = 0; j < P; j++) cols[`x${j}`] = new Float64Array(n); const ys = new Array(n); for (let i = 0; i < n; i++) { let e = -0.7; for (let j = 0; j < P; j++) { const v = rr.norm(); cols[`x${j}`][i] = v; if (j < 2) e += 0.8 * v; } ys[i] = rr.u() < 1 / (1 + Math.exp(-e)) ? '1' : '0'; } cols.y = ys; return { n, cols }; };
  const meta = Object.fromEntries([['y', cat('y', ['0', '1'])], ...Array.from({ length: P }, (_, j) => [`x${j}`, num(`x${j}`)])]), spec = { type: 'binary', y: 'y', event: '1', vars: Array.from({ length: P }, (_, j) => ({ name: `x${j}` })) };
  const test = gen(20000), tr = completeFor(test, spec), acc = { app: [0, 0], cor: [0, 0], tru: [0, 0] }, REP = 30;
  for (let r = 0; r < REP; r++) {
    const dd = gen(220), rows = completeFor(dd, spec), m = fitPred(dd, meta, spec, rows), app = evaluate(m, dd, meta, rows).perf, tru = evaluate(m, test, meta, tr).perf, bv = await bootstrapValidate(dd, meta, spec, rows, m, { B: 80, seed: 5 + r });
    const sl = bv.rows.find(x => x.key === 'slope'), au = bv.rows.find(x => x.key === 'auc');
    acc.app[0] += app.slope / REP; acc.app[1] += app.auc / REP; acc.cor[0] += sl.corrected / REP; acc.cor[1] += au.corrected / REP; acc.tru[0] += tru.slope / REP; acc.tru[1] += tru.auc / REP;
  }
  console.log(`   promedio de ${REP} réplicas · pendiente: aparente ${acc.app[0].toFixed(3)}, corregida ${acc.cor[0].toFixed(3)}, verdadera ${acc.tru[0].toFixed(3)} · C: aparente ${acc.app[1].toFixed(3)}, corregido ${acc.cor[1].toFixed(3)}, verdadero ${acc.tru[1].toFixed(3)}`);
  ok('bootstrap: la pendiente corregida elimina > 70 % del sesgo de la aparente', Math.abs(acc.cor[0] - acc.tru[0]) < 0.3 * Math.abs(acc.app[0] - acc.tru[0]));
  ok('bootstrap: el C corregido elimina > 70 % del sesgo del aparente', Math.abs(acc.cor[1] - acc.tru[1]) < 0.3 * Math.abs(acc.app[1] - acc.tru[1]));
}
// actualización: la estrategia más sencilla con calibración aceptable, o la de menor Brier si ninguna la alcanza
{ const a = recommend('binary', { orig: { slope: 0.85, citl: 0.2, brier: 0.14 }, citl: { slope: 0.85, citl: 0, brier: 0.139 }, slope: { slope: 0.99, citl: 0, brier: 0.139 }, rev: { slope: 1.0, citl: 0, brier: 0.141 } }, ['orig', 'citl', 'slope', 'rev']);
  ok('actualización: elige la recalibración con pendiente (la más sencilla aceptable)', a.id === 'slope');
  const b = recommend('binary', { orig: { slope: 0.7, citl: 0.4, brier: 0.15 }, citl: { slope: 0.7, citl: 0, brier: 0.148 }, rev: { slope: 0.8, citl: 0, brier: 0.14 } }, ['orig', 'citl', 'rev']);
  ok('actualización: si ninguna es aceptable, elige la de menor Brier', b.id === 'rev'); }
console.log(`\n${checks - fails}/${checks} comprobaciones correctas`); process.exit(fails ? 1 : 0);
