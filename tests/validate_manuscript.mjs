// Tabla 1 y regresión cruda/ajustada frente a statsmodels y SciPy; calculadora descargable frente al motor; scripts sin figuras ni enlaces.
import { readFileSync } from 'node:fs';
import { crudeAdjusted, groupColumn } from '../js/analysis.js';
import { table1 } from '../js/report.js';
import { inferMeta, typedData } from '../js/data.js';
import { parseCSV } from '../js/io.js';
import { fitPred, predictLPs, riskFromLP, completeFor, materialize } from '../js/predmodel.js';
import { calcRisk, calcPageHTML } from '../js/calcpage.js';
import { pythonScript } from '../js/pycode.js';
import { simplifyPath } from '../js/simplify.js';
import { riskDistPlot, thresholdPlot, decilePlot, importancePlot, pathPlot } from '../js/pplots.js';

const ref = JSON.parse(readFileSync(new URL('reference_ua.json', import.meta.url)));
let fails = 0, checks = 0; const ok = (n, c, x = '') => { checks++; if (!c) { fails++; console.log('FAIL', n, x); } else console.log('ok  ', n); };
const near = (a, b, t = 1e-6) => Math.abs(a - b) <= t * Math.max(1, Math.abs(b));
const rows = parseCSV(ref.csv), meta = inferMeta(rows, ['']); meta.y.event = 'Yes'; const ds = typedData(rows, meta, ['']);
const R = crudeAdjusted(ds, meta, { type: 'binary', vars: ['x1', 'x2', 'g'], y: 'y', event: 'Yes' }), eff = R.rows.filter(r => !r.header);
ok('regresión ajustada (OR) = statsmodels', eff.every((r, i) => near(r.adj.est, ref.adj[i], 1e-5)), eff.map(r => r.adj.est));
ok('regresión cruda (OR) = statsmodels', eff.every((r, i) => near(r.crude.est, ref.crude[i], 1e-5)));
ok('valores p ajustados = statsmodels', eff.every((r, i) => near(r.adj.p, ref.adj_p[i], 1e-4)));
const g = groupColumn(ds, meta, { type: 'binary', y: 'y' }), T = table1(g.ds, g.meta, ['x1', 'g'], g.group, { lang: 'en', nonparam: true }), pRow = T.rows.filter(r => r.bold).map(r => r.cells[r.cells.length - 2]);
const num = s => (s.startsWith('<') ? 0 : parseFloat(s));
ok('Tabla 1: Mann-Whitney = SciPy', Math.abs(num(pRow[0]) - ref.mw_p) < 1e-3, [pRow[0], ref.mw_p]);
ok('Tabla 1: χ² de Pearson = SciPy', Math.abs(num(pRow[1]) - ref.chi_p) < 1e-3, [pRow[1], ref.chi_p]);
ok('Tabla 1 en inglés', T.head.includes('p-value') && T.rows[0].cells[0].includes('median (IQR)'));
// calculadora: replica el motor (spline, logaritmo, categórica)
{ const n = 500, r0 = (() => { let a = 7; return () => ((a = (a * 1664525 + 1013904223) >>> 0) / 4294967296); })();
  const cols = { age: Float64Array.from({ length: n }, () => 40 + 40 * r0()), u: Float64Array.from({ length: n }, () => Math.exp(2 + 2 * r0())), s: Array.from({ length: n }, () => (r0() < 0.5 ? 'F' : 'M')) };
  cols.y = Array.from({ length: n }, (_, i) => (r0() < 1 / (1 + Math.exp(-(-6 + 0.07 * cols.age[i] + 0.3 * Math.log(cols.u[i]) + (cols.s[i] === 'M' ? 0.5 : 0)))) ? '1' : '0'));
  const md = { age: { name: 'age', type: 'num', levels: [], ref: null }, u: { name: 'u', type: 'num', levels: [], ref: null }, s: { name: 's', type: 'cat', levels: ['F', 'M'], ref: 'F' }, y: { name: 'y', type: 'cat', levels: ['0', '1'], ref: '0', event: '1' } };
  const spec = { type: 'binary', y: 'y', event: '1', vars: [{ name: 'age', form: 'rcs4' }, { name: 'u', form: 'log' }, { name: 's', form: 'lin' }], penalty: 'none' }, d0 = { n, cols }, a = materialize(spec, d0, md), rws = completeFor(a.ds, a.spec), m = fitPred(a.ds, a.meta, a.spec, rws);
  const pr = predictLPs(m, a.ds, a.meta, rws.slice(0, 20)), rk = riskFromLP(m, pr.lp, null);
  const exp = { type: m.type, intercept: m.intercept, columns: m.cols.map((c, j) => ({ name: c.name, variable: c.v, level: c.level ?? null, spline: c.spline ?? null, coef: m.beta[j] })), knots: Object.fromEntries(m.info.vars.filter(v => v.knots).map(v => [v.name, v.knots])), levels: Object.fromEntries(m.info.vars.filter(v => v.kind === 'cat').map(v => [v.name, v.levels])) };
  let dm = 0; for (let i = 0; i < 20; i++) dm = Math.max(dm, Math.abs(calcRisk(exp, { age: cols.age[rws[i]], u: cols.u[rws[i]], s: cols.s[rws[i]] }) - rk[i]));
  ok(`calculadora = motor (dif. máx. ${dm.toExponential(1)})`, dm < 1e-9);
  const html = calcPageHTML(exp, { title: 't', outcome: 'y', vars: [{ name: 'age', kind: 'num' }], n, date: '2026-01-01' }); ok('calculadora HTML autónoma, sin URL ni GitHub', !/https?:\/\/|github/i.test(html) && html.includes('calcRisk'));
  const sm = await simplifyPath(a.ds, a.meta, a.spec, rws, m, { B: 8, seed: 3, tol: 0.02 }); ok('simplificación: orden de importancia y k* válido', sm.order.length === 3 && sm.kStar >= 1 && sm.kStar <= 3 && sm.rows.length === 3);
}
const py = pythonScript({ data_file: 'd.csv', na_values: ['NA'], type: 'binary', y: 'y', event: 'Yes', penalty: 'none', B: 50, seed: 1, predictors: [{ name: 'x1', type: 'num', form: 'lin' }], predictors_raw: [{ name: 'x1', type: 'num' }], simplify: { run: false, B: 50, tol: 0.01, selected: null }, meta: { n: 400, events: 100, date: '2026-01-01' } });
ok('script Python de PREDIC: sin figuras ni GitHub; incluye bootstrap, Tabla 1 y simplificación', !/matplotlib|plt\.|github/i.test(py) && ['bootstrap_optimism', 'table1', 'crude_adjusted', 'simplify_path'].every(f => py.includes(f)));
const p = [0.1, 0.3, 0.6, 0.2, 0.8, 0.05, 0.4, 0.7], y = [0, 0, 1, 0, 1, 0, 1, 1];
ok('gráficos nuevos: sin NaN y en inglés', [riskDistPlot({ pred: p, y }), thresholdPlot({ pred: p, y }), decilePlot({ groups: [{ pred: 0.1, obs: 0.1, lo: 0.05, hi: 0.2 }, { pred: 0.5, obs: 0.6, lo: 0.4, hi: 0.8 }] }), importancePlot({ items: [{ label: 'a', value: 5 }, { label: 'b', value: 1 }] }), pathPlot({ rows: [{ k: 1, apparent: 0.7, corrected: 0.69 }, { k: 2, apparent: 0.8, corrected: 0.78 }], kStar: 2, tol: 0.01 })].every(s => !/NaN|undefined/.test(s) && !/Riesgo|Sensibilidad|Calibración/.test(s)));
console.log(`\n${checks - fails}/${checks} comprobaciones correctas`); process.exit(fails ? 1 : 0);
