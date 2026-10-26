// Eliminación hacia atrás por AIC, BIC y QIC frente a statsmodels (tests/reference_sel.json), selección repetida en el bootstrap y C de Harrell continuo.
import { readFileSync } from 'node:fs';
import { inferMeta, typedData } from '../js/data.js';
import { parseCSV } from '../js/io.js';
import { selectBackward, fitSelected, fitPred, completeFor, bootstrapValidate, evaluate, cindex } from '../js/predmodel.js';
const ref = JSON.parse(readFileSync(new URL('reference_sel.json', import.meta.url)));
const rows = parseCSV(readFileSync(new URL('../examples/erc_prediccion_desarrollo.csv', import.meta.url), 'utf8')), meta = inferMeta(rows);
meta['Falla renal a 2 años'].event = 'Sí'; const ds = typedData(rows, meta);
const spec = { type: 'binary', y: 'Falla renal a 2 años', event: 'Sí', vars: ref.vars.map(name => ({ name })) }, rr = completeFor(ds, spec);
let fails = 0; const ok = (n, c, x = '') => { if (!c) { fails++; console.log('FAIL', n, x); } else console.log('ok  ', n); };
for (const c of ['aic', 'bic', 'qic']) {
  const s = selectBackward(ds, meta, spec, rr, { crit: c });
  ok(`hacia atrás por ${c.toUpperCase()}: variables retenidas = statsmodels`, JSON.stringify(s.vars.map(v => v.name).sort()) === JSON.stringify([...ref[c].kept].sort()), s.vars.map(v => v.name));
  ok(`${c.toUpperCase()} final = statsmodels`, Math.abs(s.final - ref[c].final) < 1e-4 * Math.abs(ref[c].final), `${s.final} vs ${ref[c].final}`);
}
const m = fitSelected(ds, meta, spec, rr, {}, 'aic'); ok('fitSelected deja el modelo con las variables elegidas', m.spec.vars.length === 6 && !!m.selection);
const B = await bootstrapValidate(ds, meta, spec, rr, m, { B: 20, seed: 5, calib: false, selection: 'aic' });
ok('bootstrap con selección devuelve frecuencias de inclusión', B.selFreq.length === 8 && B.selFreq.every(f => f.freq >= 0 && f.freq <= 1) && B.B >= 15, JSON.stringify(B.selFreq));
ok('Edad se retiene en casi todas las réplicas', B.selFreq.find(f => f.name === 'Edad (años)').freq > 0.8);
// continuo: C de Harrell = proporción de parejas concordantes (cálculo directo O(n²))
const n = 300, x = Float64Array.from({ length: n }, (_, i) => Math.sin(i) * 3), y = Float64Array.from(x, (v, i) => v + Math.cos(i * 1.7) * 2);
let conc = 0, tot = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { if (y[i] === y[j]) continue; tot++; if ((x[i] - x[j]) * (y[i] - y[j]) > 0) conc++; else if (x[i] === x[j]) conc += 0.5; }
ok('C de Harrell continuo = método directo', Math.abs(cindex(Float64Array.from(x, v => -v), y, new Float64Array(n).fill(1)).c - conc / tot) < 1e-9, `${cindex(Float64Array.from(x, v => -v), y, new Float64Array(n).fill(1)).c} vs ${conc / tot}`);
console.log(fails ? `${fails} FALLOS` : 'Todo correcto'); process.exit(fails ? 1 : 0);
