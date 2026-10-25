// Pestañas 5 y 6: validación externa (modelo ajustado o ecuación publicada), recalibración y utilidad clínica (curva de decisión).
import { $, $$, esc, f2, fp, pct, ci, table, howto, card, state, errBox } from './state.js?v=2026.10.25-8';
import { readFile } from './io.js?v=2026.10.25-8';
import { inferMeta, typedData, NA_DEFAULT } from './data.js?v=2026.10.25-8';
import { evaluate, recalibrate, calibGroups, calibSurvival, lowess, dcaBinary, dcaSurvival, evalEquation, completeFor, tick, perfBinary, perfSurvival, perfContinuous, eventVec, cindex, materialize, H0at } from './predmodel.js?v=2026.10.25-8';
import { valBinary } from './samplesize.js?v=2026.10.25-8';
import { calibrationPlot, scatterCalib, rocPlot, rocPoints, dcaPlot } from './pplots.js?v=2026.10.25-8';
import { figWrap, bindFigs } from './figs.js?v=2026.10.25-8';
import { qnorm } from './dist.js?v=2026.10.25-8';
import { STRATEGIES, applyUpdate, extensionCandidates, cvCompare, recommend } from './updating.js?v=2026.10.25-8';
import { rng } from './dist.js?v=2026.10.25-8';

const z = qnorm(0.975);
const naCodes = () => [...NA_DEFAULT, ...$('#na-codes').value.split(',').map(s => s.trim()).filter(Boolean)];

// ---------------------------------------------------------------- carga de la base de validación
async function loadVal(file) {
  const { sheets } = await readFile(file), name = Object.keys(sheets)[0], raw = sheets[name].map(r => ({ ...r }));
  if (!raw.length) throw new Error('El archivo no tiene filas.');
  state.val = { name: file.name, raw, meta: inferMeta(raw, naCodes()) };
  $('#v-info').innerHTML = `<span class="pill">✓ ${esc(file.name)}: ${raw.length} filas × ${Object.keys(raw[0]).length} columnas</span>`;
  if (state.vmode === 'eq') fillEqSelects();
}
// tipado de la base de validación con los mismos tipos/referencias del desarrollo y con las variables transformadas
function valDataset() {
  const v = state.val, active = {};
  for (const [k, m] of Object.entries(state.meta)) { if (m.type === 'skip') continue; if (!(k in v.raw[0])) continue; active[k] = m; }
  const ds0 = typedData(v.raw, active, naCodes());
  const miss = [...new Set([...state.spec.vars.map(x => x.name), ...(state.spec.type === 'survival' ? [state.spec.time, state.spec.status] : [state.spec.y])])].filter(k => !(k in v.raw[0]));
  if (miss.length) throw new Error(`La base de validación no contiene: ${miss.join(', ')}.`);
  return materialize(state.spec, ds0, state.meta);
}
const bootC = (lp, time, status, B = 150, seed = 3) => { const R = rng(seed), n = lp.length, cs = []; for (let b = 0; b < B; b++) { const ix = Array.from({ length: n }, () => R.int(n)), r = cindex(Float64Array.from(ix, i => lp[i]), Float64Array.from(ix, i => time[i]), Float64Array.from(ix, i => status[i])).c; if (isFinite(r)) cs.push(r); } cs.sort((a, b) => a - b); return [cs[Math.floor(0.025 * cs.length)], cs[Math.ceil(0.975 * cs.length) - 1]]; };

// ---------------------------------------------------------------- resultados de validación
function perfTable(type, pf, dev, extra = {}) {
  const dc = (k) => dev?.rows?.find(r => r.key === k)?.corrected;
  if (type === 'binary') return table(['Medida', 'Validación (IC 95 %)', 'Desarrollo (corregido)', 'Ideal'], [
    ['Estadístico C (AUC)', ci(pf.auc, pf.aucLo, pf.aucHi, 3), f2(dc('auc'), 3), ''],
    ['Pendiente de calibración', ci(pf.slope, pf.slope - z * pf.slopeSE, pf.slope + z * pf.slopeSE, 3), f2(dc('slope'), 3), '1'],
    ['Calibración en general (CITL)', ci(pf.citl, pf.citl - z * pf.citlSE, pf.citl + z * pf.citlSE, 3), f2(dc('citl'), 3), '0'],
    ['Observados/esperados (O/E)', ci(pf.oe, Math.exp(Math.log(pf.oe) - z * Math.sqrt(1 / pf.events - 1 / pf.n)), Math.exp(Math.log(pf.oe) + z * Math.sqrt(1 / pf.events - 1 / pf.n)), 3), '', '1'],
    ['Brier (escalado)', `${f2(pf.brier, 4)} (${f2(pf.brierScaled, 3)})`, f2(dc('brier'), 4), ''], ['R² de Cox-Snell', f2(pf.r2cs, 3), f2(dc('r2cs'), 3), '']]);
  if (type === 'survival') return table(['Medida', 'Validación (IC 95 %)', 'Desarrollo (corregido)', 'Ideal'], [
    ['Índice C de Harrell', extra.cci ? ci(pf.cindex, extra.cci[0], extra.cci[1], 3) : f2(pf.cindex, 3), f2(dc('cindex'), 3), ''],
    ['Pendiente de calibración (PI)', ci(pf.slope, pf.slope - z * pf.slopeSE, pf.slope + z * pf.slopeSE, 3), f2(dc('slope'), 3), '1'],
    [`Riesgo observado / esperado a ${pf.horizon}`, `${pct(pf.observed)} / ${pct(pf.expected)} (O/E = ${f2(pf.oe, 2)})`, '', '1'], [`Brier a ${pf.horizon} (ponderado por censura)`, f2(pf.brier, 4), '', ''], ['R² de Cox-Snell', f2(pf.r2cs, 3), f2(dc('r2cs'), 3), '']]);
  return table(['Medida', 'Validación', 'Desarrollo (corregido)', 'Ideal'], [['R²', f2(pf.r2, 3), f2(dc('r2'), 3), ''], ['Pendiente de calibración', ci(pf.slope, pf.slope - z * pf.slopeSE, pf.slope + z * pf.slopeSE, 3), f2(dc('slope'), 3), '1'], ['CITL (media observada − predicha)', f2(pf.citl, 3), '', '0'], ['RMSE / MAE', `${f2(pf.rmse, 3)} / ${f2(pf.mae, 3)}`, f2(dc('rmse'), 3), '']]);
}
function sdOf(a) { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); }
const meanOf = a => a.reduce((s, v) => s + v, 0) / a.length;

function showVal(ev, m, label, { equation = false } = {}) {
  const type = m.type, pf = ev.perf, out = $('#val-out'), dev = state.boot;
  const devLP = state.dev?.app?.lp, cm = devLP && ev.lp ? `<p class="sub"><b>Comparación de las poblaciones.</b> Predictor lineal en desarrollo: media ${f2(meanOf(devLP), 2)}, DE ${f2(sdOf(devLP), 2)}; en validación: media ${f2(meanOf(ev.lp), 2)}, DE ${f2(sdOf(ev.lp), 2)}. ${sdOf(ev.lp) > sdOf(devLP) * 1.15 ? 'Una población más heterogénea facilita la discriminación (C mayor).' : sdOf(ev.lp) < sdOf(devLP) * 0.85 ? 'Una población más homogénea reduce la discriminación (C menor) aunque el modelo sea correcto.' : 'Dispersión parecida del predictor lineal.'}</p>` : '';
  let plots = '', adequacy = '', recal = '';
  if (type === 'binary') {
    const g = calibGroups(ev.p, ev.y, 10), curve = lowess(ev.p, ev.y, 2 / 3, 40).map(([a, b]) => [a, Math.min(1, Math.max(0, b))]);
    plots = figWrap(calibrationPlot({ groups: g, curve, pred: ev.p, y: ev.y, title: `Calibration · ${label}`, note: `Slope ${f2(pf.slope, 2)} · CITL ${f2(pf.citl, 2)} · O/E ${f2(pf.oe, 2)}` }), 'calibracion_validacion') + figWrap(rocPlot(rocPoints(ev.p, ev.y), `C = ${f2(pf.auc, 3)}`, 'ROC in validation'), 'roc_validacion');
    try { const need = valBinary({ prev: pf.prev, cstat: pf.auc, seOE: 0.1, seSlope: 0.0765, seCstat: 0.0255 }); adequacy = `<p class="sub"><b>¿Es suficiente la muestra de validación?</b> Tiene ${pf.n} pacientes y ${pf.events} eventos. Para IC con ancho ≤ 0.3 en la pendiente, ≤ 0.1 en el C y ±0.2 en ln(O/E) (Riley et al. <i>BMJ</i> 2024) con prevalencia ${pct(pf.prev)} y C = ${f2(pf.auc, 2)} se requieren ≈ <b>${need.n.toLocaleString('es')}</b> pacientes${pf.n >= need.n ? ': <span class="ok">suficiente</span>.' : ': <span class="bad">los intervalos serán más anchos que el objetivo.</span>'}</p>`; } catch { /* sin cálculo */ }
  } else if (type === 'survival') {
    const g = calibSurvival(ev.p, ev.time, ev.status, pf.horizon, 5), mx = Math.min(1, Math.max(0.05, ...g.map(r => Math.max(r.pred, r.hi))) * 1.1);
    plots = figWrap(calibrationPlot({ groups: g, curve: null, pred: ev.p, title: `Calibration at ${pf.horizon} · ${label}`, note: 'Observed = Kaplan–Meier by risk quintile (95% CI)', max: mx }), 'calibracion_validacion_cox');
  } else plots = figWrap(scatterCalib({ pred: ev.p, y: ev.y, curve: lowess(ev.p, ev.y, 2 / 3, 40), title: `Observed vs predicted · ${label}` }), 'calibracion_validacion_lineal');
  if (!equation && type === 'survival') recal = `<div class="explain"><b>Recalibración.</b> Si la calibración es deficiente, puede actualizar el modelo con los datos de validación: solo el intercepto (corrige el riesgo medio) o intercepto y pendiente (corrige además el exceso de dispersión). El resultado es una versión actualizada que debe validarse otra vez.<div class="row gap mt wrap"><button class="ghost" data-rc="citl">Recalibrar intercepto</button><button class="ghost" data-rc="slope">Recalibrar intercepto y pendiente</button><button class="primary" id="v-use" ${state.recalM ? '' : 'hidden'}>Usar el modelo recalibrado como modelo final</button></div><div id="v-recal"></div></div>`;
  out.innerHTML = card(`Resultados de la validación · ${esc(label)}`, `${howto(`Rendimiento del modelo sin reajustar en ${pf.n} pacientes${pf.events != null ? ` (${pf.events} eventos)` : ''}.`, 'Compara con el rendimiento corregido por bootstrap del desarrollo: una caída del C o una pendiente &lt; 0.9 indican falta de transportabilidad.', 'La discriminación depende de la heterogeneidad de la población; la calibración (pendiente, CITL) es la medida clave para usar riesgos absolutos.', ['Qué se ve', 'Para qué', 'Cómo interpretar'])}${perfTable(type, pf, dev, ev.extra)}${cm}${adequacy}<div class="plots">${plots}</div>${recal}`);
  if (!equation && type !== 'survival') { out.insertAdjacentHTML('beforeend', updateCard(m)); bindUpdate(); }
  bindFigs(out);
  $$('[data-rc]', out).forEach(b => b.addEventListener('click', () => doRecal(b.dataset.rc)));
  state.valEv = ev; state.valLabel = label;
}
// ---------------------------------------------------------------- actualización del modelo (recalibración, revisión, extensión)
function updateCard(m) {
  const cands = extensionCandidates(m, state.meta, state.valPrep.ds), box = (id, lab, on) => `<label class="inline"><input type="checkbox" data-up="${id}"${on ? ' checked' : ''}${id === 'orig' ? ' disabled' : ''}> ${esc(lab)}</label>`;
  return `<div class="card" id="up-card"><h2>Actualizar el modelo con los datos de validación</h2>${howto('Cuatro maneras de adaptar el modelo al nuevo entorno, de la más sencilla a la más flexible: corregir el nivel de riesgo (intercepto), corregir también la escala (pendiente), reestimar todos los coeficientes (revisión) o añadir un marcador nuevo (extensión).', 'Si el modelo discrimina bien pero se calibra mal, basta recalibrar. Cuanto más se reestima, más eventos hacen falta y más se pierde del modelo original.', 'Cada estrategia se evalúa con validación cruzada dentro de la base de validación (se actualiza con 4/5 y se prueba en el 1/5 restante, 3 repeticiones), no con los mismos datos con los que se ajustó. Elija la más sencilla que deje la pendiente en 0.9–1.1 y el intercepto cerca de 0.')}
    <div class="row gap wrap">${STRATEGIES.map(([id, lab]) => box(id, lab, id !== 'ext')).join('')}</div>
    <div class="row gap wrap mt"><label>Marcador nuevo (solo para la extensión) <select id="up-extra">${cands.length ? cands.map(c => `<option>${esc(c)}</option>`).join('') : '<option value="">(no hay variables adicionales)</option>'}</select></label></div>
    <div class="row gap mt wrap"><button class="primary" id="up-run">Comparar estrategias</button><span id="up-msg" class="muted"></span></div><div id="up-out"></div></div>`;
}
function bindUpdate() {
  const m = state.final, bin = m.type === 'binary', num = (v, d = 3) => (isFinite(v) ? f2(v, d) : '–');
  $('#up-run').addEventListener('click', async () => {
    const msg = $('#up-msg'), outEl = $('#up-out'); msg.textContent = 'Comparando…'; outEl.innerHTML = ''; await tick();
    try {
      const ids = $$('[data-up]').filter(c => c.checked || c.dataset.up === 'orig').map(c => c.dataset.up), extra = ids.includes('ext') ? $('#up-extra').value : null;
      if (ids.includes('ext') && !extra) throw new Error('No hay un marcador adicional para la extensión: desmarque esa estrategia.');
      const { prep, rows } = { prep: state.valPrep, rows: state.valRows }, { res, n } = cvCompare({ ids, m, ds: prep.ds, meta: prep.meta, rows, extra }), rec = recommend(m.type, res, ids);
      const lab = Object.fromEntries(STRATEGIES.map(([id, l]) => [id, l])), par = id => (id === 'rev' ? m.cols.length + 1 : id === 'ext' ? m.cols.length + 2 : { orig: 0, citl: 1, slope: 2 }[id]);
      const head = bin ? ['Estrategia', 'Parámetros reestimados', 'C (AUC)', 'Pendiente', 'CITL', 'O/E', 'Brier'] : ['Estrategia', 'Parámetros reestimados', 'R²', 'Pendiente', 'CITL', 'RMSE'];
      const rowsT = ids.map(id => { const r = res[id], b = id === rec?.id ? '<b>' : '', e = id === rec?.id ? '</b>' : ''; return r.err ? [esc(lab[id]), par(id), `<span class="muted">${esc(r.err)}</span>`, '', '', '', ''].slice(0, head.length) : [`${b}${esc(lab[id])}${e}`, par(id), ...(bin ? [num(r.auc), num(r.slope), num(r.citl), num(r.oe), num(r.brier, 4)] : [num(r.r2), num(r.slope), num(r.citl), num(r.rmse)])]; });
      const adopt = ids.filter(id => id !== 'ext' && !res[id].err);
      outEl.innerHTML = `<p class="sub">Rendimiento por validación cruzada (5 pliegues × 3 repeticiones) en ${n} pacientes de la base de validación.</p>${table(head, rowsT)}
        ${rec ? `<p><b>Sugerencia:</b> ${esc(lab[rec.id])} — ${esc(rec.why)}.</p>` : ''}${ids.includes('ext') && res.ext && !res.ext.err ? `<p class="sub">La extensión añade «${esc(extra)}»: compare su Brier/AUC con la revisión para ver si el marcador aporta. Para adoptarla, añada la variable en el paso 3 y reajuste.</p>` : ''}
        <div class="row gap wrap"><label>Usar como modelo final <select id="up-pick">${adopt.map(id => `<option value="${id}"${id === rec?.id ? ' selected' : ''}>${esc(lab[id])}</option>`).join('')}</select></label><button class="ghost" id="up-use">Usar</button></div><p class="sub" id="up-done"></p>`;
      $('#up-use').addEventListener('click', () => { try { const id = $('#up-pick').value, mk = applyUpdate(id, state.final, prep.ds, prep.meta, rows); state.final = mk; state.shrunk = null; state.updated = { id, label: lab[id], n: rows.length }; $('#up-done').textContent = id === 'orig' ? 'Se mantiene el modelo original.' : `Hecho: «${lab[id]}» es ahora el modelo final (se usará en utilidad clínica y reporte). Se estimó con las ${rows.length} filas completas de la base de validación; valídelo de nuevo en datos independientes.`; } catch (e) { $('#up-done').innerHTML = errBox(e); } });
      msg.textContent = '';
    } catch (e) { msg.textContent = ''; outEl.innerHTML = errBox(e); }
  });
}
function doRecal(kind) {
  try {
    const { ds, meta } = state.valPrep, rows = state.valRows, m = state.final, rc = recalibrate(m, ds, meta, rows, kind), ev = evaluate(rc, ds, meta, rows, { horizon: state.spec.horizon });
    state.recalM = rc; const pf = ev.perf;
    $('#v-recal').innerHTML = `<p class="sub">Modelo ${kind === 'citl' ? 'con intercepto' : 'con intercepto y pendiente'} recalibrado${rc.recalSlope ? ` (pendiente de recalibración = ${f2(rc.recalSlope, 3)})` : ''}. Rendimiento <b>aparente</b> en la base de validación (optimista):</p>${table(['Medida', 'Valor'], type2rows(rc.type, pf))}`;
    $('#v-use').hidden = false; $('#v-use').onclick = () => { state.final = rc; state.shrunk = null; alert('El modelo recalibrado es ahora el modelo final (se usará en utilidad clínica y reporte).'); };
  } catch (e) { $('#v-recal').innerHTML = errBox(e); }
}
const type2rows = (t, pf) => (t === 'binary' ? [['Pendiente de calibración', f2(pf.slope, 3)], ['CITL', f2(pf.citl, 3)], ['O/E', f2(pf.oe, 3)], ['C', f2(pf.auc, 3)]] : t === 'survival' ? [['Pendiente', f2(pf.slope, 3)], ['O/E a ' + pf.horizon, f2(pf.oe, 3)], ['C', f2(pf.cindex, 3)]] : [['Pendiente', f2(pf.slope, 3)], ['CITL', f2(pf.citl, 3)], ['R²', f2(pf.r2, 3)]]);

async function runDataMode() {
  if (!state.val) throw new Error('Cargue primero la base de validación.');
  const prep = valDataset(), m = state.final, rows = Array.from({ length: prep.ds.n }, (_, i) => i), r2 = completeFor(prep.ds, prep.spec);
  const ev = evaluate(m, prep.ds, prep.meta, r2, { horizon: state.spec.horizon });
  if (Object.keys(ev.unseen).length) $('#v-msg').textContent = `Aviso: categorías no vistas en el desarrollo (excluidas): ${JSON.stringify(ev.unseen)}`;
  if (m.type === 'survival') ev.extra = { cci: bootC(ev.lp, ev.time, ev.status) };
  state.valPrep = prep; state.valRows = r2; state.recalM = null; state.valDs = { ev, prep };
  showVal(ev, m, state.val.name);
  $('#v-msg').textContent = `${ev.used.length} de ${state.val.raw.length} pacientes con datos completos.`;
}

// ---------------------------------------------------------------- ecuación publicada
function drawEq() {
  const T = $('#eq-table'); T.innerHTML = `<thead><tr><th>Columna de la base</th><th>Coeficiente</th><th></th></tr></thead><tbody>${state.eqTerms.map((t, i) => `<tr><td><input data-eq="${i}" data-k="name" value="${esc(t.name)}" list="eq-cols" size="28"></td><td><input data-eq="${i}" data-k="coef" value="${esc(t.coef)}" type="number" step="any"></td><td><button class="x" data-eqdel="${i}">×</button></td></tr>`).join('')}</tbody><datalist id="eq-cols">${Object.keys(state.val?.raw?.[0] || {}).map(c => `<option>${esc(c)}</option>`).join('')}</datalist>`;
  $$('input', T).forEach(i => i.addEventListener('input', () => (state.eqTerms[+i.dataset.eq][i.dataset.k] = i.value))); $$('[data-eqdel]', T).forEach(b => b.addEventListener('click', () => { state.eqTerms.splice(+b.dataset.eqdel, 1); drawEq(); }));
}
function fillEqSelects() {
  if (!$('#eq-out-sel')) $('#val-eq').insertAdjacentHTML('beforeend', '<div class="form mt" id="eq-out-sel"></div>');
  const cols = Object.keys(state.val?.raw?.[0] || {}), t = $('#eq-type').value, m = state.val?.meta || {}, lv = v => m[v]?.levels || [];
  $('#eq-out-sel').innerHTML = t === 'binary' ? `<label>Desenlace <select id="eq-y">${cols.map(c => opt0(c)).join('')}</select></label><label>Categoría que es el evento <select id="eq-ev"></select></label>` : `<label>Tiempo <select id="eq-t">${cols.map(c => opt0(c)).join('')}</select></label><label>Indicador de evento <select id="eq-st">${cols.map(c => opt0(c)).join('')}</select></label><label>Categoría que es el evento <select id="eq-ev"></select></label>`;
  const sync = () => { const v = $(t === 'binary' ? '#eq-y' : '#eq-st').value; $('#eq-ev').innerHTML = lv(v).map(l => `<option${l === m[v]?.event ? ' selected' : ''}>${esc(l)}</option>`).join(''); };
  $(t === 'binary' ? '#eq-y' : '#eq-st').addEventListener('change', sync); sync(); drawEq();
}
const opt0 = c => `<option>${esc(c)}</option>`;
function runEqMode() {
  if (!state.val) throw new Error('Cargue la base de validación.');
  const terms = state.eqTerms.filter(t => t.name && t.coef !== '').map(t => ({ name: t.name, coef: parseFloat(t.coef) })); if (!terms.length) throw new Error('Añada al menos un término de la ecuación.');
  const type = $('#eq-type').value, m0 = state.val.meta, act = Object.fromEntries(Object.entries(m0).filter(([k]) => terms.some(t => t.name === k) || k === (type === 'binary' ? $('#eq-y').value : $('#eq-t').value) || k === (type === 'binary' ? '' : $('#eq-st').value)));
  for (const t of terms) if (act[t.name]) act[t.name] = { ...act[t.name], type: 'num' };
  const ds = typedData(state.val.raw, act, naCodes()), rows = Array.from({ length: ds.n }, (_, i) => i), eq = { type, intercept: parseFloat($('#eq-int').value) || 0, terms }, { lp, used } = evalEquation(eq, ds, rows);
  if (type === 'binary') {
    const y = eventVec(ds, { y: $('#eq-y').value, event: $('#eq-ev').value }, used), p = Float64Array.from(lp, v => 1 / (1 + Math.exp(-v)));
    showVal({ perf: perfBinary(p, y), lp, p, y, used }, { type: 'binary' }, 'ecuación publicada', { equation: true });
  } else {
    const s0 = parseFloat($('#eq-s0').value), ref = parseFloat($('#eq-ref').value) || 0, h = parseFloat($('#eq-h').value), H = -Math.log(s0), status = Float64Array.from(used, i => (String(ds.cols[$('#eq-st').value][i]) === $('#eq-ev').value ? 1 : 0)), time = Float64Array.from(used, i => +ds.cols[$('#eq-t').value][i]);
    const risk = Float64Array.from(lp, v => 1 - Math.exp(-H * Math.exp(v - ref))), pf = perfSurvival(lp, time, status, { base: { t: [0, h], H: [0, H] }, lpMean: ref, horizon: h });
    showVal({ perf: pf, lp, p: risk, time, status, used, extra: { cci: bootC(lp, time, status) } }, { type: 'survival' }, 'ecuación publicada', { equation: true });
  }
  $('#v-msg').textContent = `${used.length} de ${state.val.raw.length} pacientes con datos completos.`;
}

export function initValidation() {
  $('#vfile').addEventListener('change', e => e.target.files[0] && loadVal(e.target.files[0]).catch(err => ($('#v-info').innerHTML = errBox(err))));
  $('#vfile2').addEventListener('change', e => e.target.files[0] && loadVal(e.target.files[0]).then(() => drawEq()).catch(err => ($('#v-info').innerHTML = errBox(err))));
  $('#vex').addEventListener('click', async e => { e.preventDefault(); try { const t = await (await fetch('examples/erc_prediccion_validacion.csv')).text(); await loadVal(new File([t], 'erc_prediccion_validacion.csv')); } catch (err) { $('#v-info').innerHTML = errBox(err); } });
  $$('[data-vm]').forEach(b => b.addEventListener('click', () => { state.vmode = b.dataset.vm; $$('[data-vm]').forEach(x => x.classList.toggle('active', x === b)); $('#val-data').hidden = b.dataset.vm !== 'data'; $('#val-eq').hidden = b.dataset.vm !== 'eq'; if (b.dataset.vm === 'eq') { if (state.val) fillEqSelects(); else drawEq(); } }));
  $('#eq-add').addEventListener('click', () => { state.eqTerms.push({ name: '', coef: '' }); drawEq(); });
  $('#eq-type').addEventListener('change', () => { const t = $('#eq-type').value; $$('.eqb').forEach(e => (e.hidden = t !== 'binary')); $$('.eqs').forEach(e => (e.hidden = t !== 'survival')); if (state.val) fillEqSelects(); }); $$('.eqs').forEach(e => (e.hidden = true));
  $('#v-run').addEventListener('click', async () => { $('#v-msg').textContent = 'Evaluando…'; await tick(); try { if (state.vmode === 'data') await runDataMode(); else runEqMode(); } catch (e) { $('#v-msg').textContent = ''; $('#val-out').innerHTML = card('Error', errBox(e)); } });
  $('#u-run').addEventListener('click', runDCA); drawEq();
}

// ---------------------------------------------------------------- utilidad clínica
export function renderUtilityInit() {
  const sel = $('#u-src'); if (!sel) return; const cur = sel.value;
  sel.innerHTML = `<option value="dev">Desarrollo (aparente, optimista)</option>${state.valEv ? `<option value="val">Validación: ${esc(state.valLabel)}</option>` : ''}`; if (cur && [...sel.options].some(o => o.value === cur)) sel.value = cur; else if (state.valEv) sel.value = 'val';
}
function runDCA() {
  const out = $('#u-out'); try {
    const src = $('#u-src').value, ev = src === 'val' ? state.valEv : state.dev.app, spec = state.spec, type = state.final.type;
    if (type === 'continuous') throw new Error('La curva de decisión requiere un desenlace binario o de supervivencia.');
    const lo = +$('#u-min').value, hi = +$('#u-max').value, pt0 = +$('#u-pt').value; if (!(lo > 0 && hi < 1 && hi > lo)) throw new Error('Umbrales no válidos (0 < mín. < máx. < 1).');
    const th = Array.from({ length: 50 }, (_, k) => lo + (hi - lo) * k / 49), h = state.valEv && src === 'val' && state.valEv.perf.horizon || spec.horizon;
    const rows = type === 'binary' ? dcaBinary(ev.p, ev.y, th) : dcaSurvival(ev.p, ev.time, ev.status, h, th), at = type === 'binary' ? dcaBinary(ev.p, ev.y, [pt0])[0] : dcaSurvival(ev.p, ev.time, ev.status, h, [pt0])[0];
    // clasificación al umbral de interés (binario)
    let cls = '';
    if (type === 'binary') { let tp = 0, fp = 0, tn = 0, fn = 0; ev.p.forEach((p, i) => { const pos = p >= pt0; if (ev.y[i]) pos ? tp++ : fn++; else pos ? fp++ : tn++; }); cls = table(['Con umbral ' + pt0, 'Valor'], [['Sensibilidad', pct(tp / (tp + fn))], ['Especificidad', pct(tn / (tn + fp))], ['Valor predictivo positivo', pct(tp / (tp + fp))], ['Valor predictivo negativo', pct(tn / (tn + fn))], ['Pacientes por encima del umbral', pct((tp + fp) / ev.p.length)]]); }
    const pts = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4].filter(x => x >= lo && x <= hi), tr = (type === 'binary' ? dcaBinary(ev.p, ev.y, pts) : dcaSurvival(ev.p, ev.time, ev.status, h, pts)).map(r => [r.pt, f2(r.model, 4), f2(r.all, 4), '0', f2(r.avoid, 1)]);
    out.innerHTML = `<div class="plots single">${figWrap(dcaPlot(rows, { title: `Decision curve${type === 'survival' ? ' at ' + h : ''} · ${src === 'val' ? 'validation' : 'development (apparent)'}` }), 'curva_decision')}</div><div class="kpis"><div class="kpi"><b>${f2(at.model, 4)}</b><span>beneficio neto del modelo (pt = ${pt0})</span></div><div class="kpi"><b>${f2(at.all, 4)}</b><span>tratar a todos</span></div><div class="kpi"><b>${f2(at.avoid, 1)}</b><span>intervenciones evitadas por 100 pacientes</span></div></div>${cls}<h3>Beneficio neto por umbral</h3>${table(['Umbral', 'Modelo', 'Tratar a todos', 'No tratar', 'Intervenciones evitadas por 100'], tr)}<p class="notes">Beneficio neto = VP/n − FP/n·pt/(1−pt) (Vickers y Elkin, 2006). En supervivencia se estima con Kaplan-Meier a ${h} en los pacientes sobre el umbral. ${src === 'dev' ? 'Con los datos de desarrollo el resultado es optimista; use la validación externa si dispone de ella.' : ''}</p>`;
    bindFigs(out);
  } catch (e) { out.innerHTML = errBox(e); }
}
