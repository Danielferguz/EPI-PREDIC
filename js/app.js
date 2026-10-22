// EPI-PREDIC · navegación, carga de datos, especificación del modelo y ejecución.
import { $, $$, esc, f2, pct, opt, errBox, state, natural, download } from './state.js?v=2026.10.22-1';
import { readFile } from './io.js?v=2026.10.22-1';
import { inferMeta, typedData, NA_DEFAULT, missCount, levelsOf } from './data.js?v=2026.10.22-1';
import { mean, sd } from './tests.js?v=2026.10.22-1';
import { completeFor, fitPred, evaluate, bootstrapValidate, columnNames, designInfo, tick, materialize } from './predmodel.js?v=2026.10.22-1';
import { initSampleSize } from './ui_ss.js?v=2026.10.22-1';
import { renderDev, bindFigure } from './ui_dev.js?v=2026.10.22-1';
import { initValidation, renderUtilityInit } from './ui_val.js?v=2026.10.22-1';
import { renderReport } from './ui_rep.js?v=2026.10.22-1';
import { bindFigs } from './figs.js?v=2026.10.22-1';
import { HELP_HTML } from './help.js?v=2026.10.22-1';

// ---------------- ejemplos (datos simulados de nefrología) ----------------
const EXAMPLES = [
  { file: 'erc_prediccion_desarrollo.csv', title: 'ERC · falla renal a 2 años (logístico)', design: 'Cohorte de desarrollo · 1800 pacientes', use: 'Modelo logístico con splines, bootstrap, calibración y DCA. Validación: ejemplo simulado en el paso 5.',
    meta: { 'Falla renal a 2 años': { event: 'Sí' } }, drop: ['ID', 'Seguimiento (años)', 'Falla renal'], spec: { type: 'binary', y: 'Falla renal a 2 años', vars: { 'Edad (años)': 'lin', Sexo: 'lin', Diabetes: 'lin', 'TFGe (mL/min/1.73 m²)': 'rcs4', 'UACR (mg/g)': 'log', 'Hemoglobina (g/dL)': 'lin', 'Bicarbonato (mmol/L)': 'lin', 'Fosfato (mg/dL)': 'lin' } } },
  { file: 'erc_prediccion_desarrollo.csv', title: 'ERC · falla renal en el tiempo (Cox)', design: 'Cohorte de desarrollo · 1800 pacientes', use: 'Modelo de Cox con riesgo a 2 años, C de Harrell, calibración por grupos y DCA con censura.',
    meta: { 'Falla renal': { event: 'Sí' } }, drop: ['ID', 'Falla renal a 2 años'], spec: { type: 'survival', time: 'Seguimiento (años)', status: 'Falla renal', horizon: 2, vars: { 'Edad (años)': 'lin', Sexo: 'lin', Diabetes: 'lin', 'TFGe (mL/min/1.73 m²)': 'rcs4', 'UACR (mg/g)': 'log', 'Hemoglobina (g/dL)': 'lin', 'Bicarbonato (mmol/L)': 'lin', 'Fosfato (mg/dL)': 'lin' } } },
  { file: 'lesion_renal_aguda.csv', title: 'Lesión renal aguda poscirugía cardiaca', design: 'Cohorte · 520 pacientes', use: 'Modelo logístico con pocos eventos: ridge/lasso y optimismo.',
    meta: { 'LRA (KDIGO)': { event: 'Sí' } }, drop: ['ID'], spec: { type: 'binary', y: 'LRA (KDIGO)', vars: { 'Edad': 'lin', Diabetes: 'lin', 'TFGe basal (mL/min/1.73 m²)': 'lin', 'Fracción de eyección (%)': 'lin', 'Tipo de cirugía': 'lin', 'Tiempo de bypass (min)': 'lin', Vasopresores: 'lin' } } },
  { file: 'hemodialisis_mortalidad.csv', title: 'Hemodiálisis · mortalidad', design: 'Cohorte · 450 pacientes', use: 'Modelo de supervivencia (Cox).',
    meta: { Fallecido: { event: 'Sí' } }, drop: ['ID'], spec: { type: 'survival', time: 'Tiempo de seguimiento (meses)', status: 'Fallecido', horizon: 12, vars: { Edad: 'lin', Diabetes: 'lin', 'Índice de Charlson': 'lin', 'Albúmina (g/dL)': 'lin', 'Hemoglobina (g/dL)': 'lin', 'Acceso vascular': 'lin' } } },
  { file: 'hemodialisis_depresion.csv', title: 'Hemodiálisis · puntaje PHQ-9', design: 'Transversal · 260 pacientes', use: 'Modelo lineal (desenlace continuo).',
    meta: {}, drop: ['ID', 'Depresión', 'Grupo de edad'], spec: { type: 'continuous', y: 'Puntaje PHQ-9', vars: { Edad: 'lin', Sexo: 'lin', Diabetes: 'lin', 'Apoyo familiar': 'lin', 'Hemoglobina (g/dL)': 'lin', 'Albúmina (g/dL)': 'lin' } } },
];

// ---------------- navegación ----------------
function showTab(t) {
  $$('.tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === t)); $$('.tab').forEach(s => s.classList.toggle('active', s.id === 'tab-' + t));
  if (t === 'modelo') renderModelForm(); if (t === 'utilidad') renderUtilityInit(); if (t === 'reporte') renderReport(); if (t === 'validacion') $('#v-info').textContent = state.val ? `Base de validación: ${state.val.name} (${state.val.raw.length} filas)` : $('#v-info').textContent;
  window.scrollTo({ top: 0 });
}
export { showTab };
$$('.tabs button').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));
document.addEventListener('click', e => { const g = e.target.closest('[data-go]'); if (g) showTab(g.dataset.go); });
$('#help').innerHTML = HELP_HTML;
export function gate() { $$('.need-data').forEach(e => (e.hidden = !!state.ds)); $$('.has-data').forEach(e => (e.hidden = !state.ds)); $$('.need-model').forEach(e => (e.hidden = !!state.final)); $$('.has-model').forEach(e => (e.hidden = !state.final)); }

// ---------------- carga ----------------
$('#examples').innerHTML = EXAMPLES.map((x, k) => `<button class="ghost" data-ex="${k}"><em>${esc(x.design)}</em><b>${esc(x.title)}</b><small>${esc(x.use)}</small></button>`).join('');
$$('[data-ex]').forEach(b => b.addEventListener('click', async () => {
  const x = EXAMPLES[+b.dataset.ex], r = await fetch('examples/' + x.file), txt = await r.text();
  await loadFile(new File([txt], x.file, { type: 'text/csv' }));
  for (const [v, o] of Object.entries(x.meta || {})) if (state.meta[v]) Object.assign(state.meta[v], o);
  for (const v of x.drop || []) if (state.meta[v]) state.meta[v].type = 'skip';
  rebuild(); state.draft = x.spec; renderVarTable(); renderModelForm();
  $$('[data-ex]').forEach(e => e.classList.toggle('active', e === b));
}));
$('#file').addEventListener('change', e => e.target.files[0] && loadFile(e.target.files[0]));
const drop = $('#drop');
['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); })); ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => e.dataTransfer.files[0] && loadFile(e.dataTransfer.files[0]));
export const naCodes = () => [...NA_DEFAULT, ...$('#na-codes').value.split(',').map(s => s.trim()).filter(Boolean)];
async function loadFile(file) {
  try {
    const { sheets } = await readFile(file), name = Object.keys(sheets)[0]; state.fname = file.name.replace(/\.[^.]+$/, ''); state.raw = sheets[name].map(r => ({ ...r }));
    if (!state.raw.length) throw new Error('El archivo no tiene filas.');
    state.meta = inferMeta(state.raw, naCodes()); Object.assign(state, { model: null, final: null, boot: null, shrunk: null, spec: null, draft: null }); rebuild();
    $('#load-msg').innerHTML = `<p class="pill">✓ ${esc(state.fname)}: ${state.raw.length} filas × ${Object.keys(state.meta).length} columnas</p>`;
    $('#dev-out').innerHTML = ''; renderVarTable(); gate();
  } catch (e) { $('#load-msg').innerHTML = errBox(e); }
}
$('#na-codes').addEventListener('change', () => state.raw.length && (state.meta = Object.assign(inferMeta(state.raw, naCodes()), {}), rebuild(), renderVarTable()));
export function rebuild() { state.ds = typedData(state.raw, Object.fromEntries(Object.entries(state.meta).filter(([, m]) => m.type !== 'skip')), naCodes()); }
const usable = (...types) => Object.values(state.meta).filter(m => m.type !== 'skip' && (!types.length || types.includes(m.type))).map(m => m.name);
function renderVarTable() {
  if (!state.ds) return; const T = $('#vartable');
  T.innerHTML = `<thead><tr><th>Variable</th><th>Tipo</th><th>Referencia</th><th>Evento (Sí/No)</th><th class="num">Perdidos</th><th>Resumen</th></tr></thead><tbody>${Object.values(state.meta).map(m => {
    const col = state.ds.cols[m.name], miss = col ? missCount(col) : 0, isCat = m.type === 'cat'; let s = '';
    if (m.type === 'num' && col) { const v = Array.from(col).filter(isFinite); s = `media ${f2(mean(v))} ± ${f2(sd(v))}; rango ${f2(Math.min(...v))}–${f2(Math.max(...v))}`; }
    else if (isCat && col) { const c = {}; col.forEach(x => x != null && (c[x] = (c[x] || 0) + 1)); s = levelsOf(m).slice(0, 6).map(l => `${l} (${c[l] || 0})`).join(' · ') + (m.levels.length > 6 ? ' …' : ''); }
    return `<tr data-v="${esc(m.name)}"><td>${esc(m.name)}</td><td><select data-k="type">${opt('num', 'Cuantitativa', m.type === 'num')}${opt('cat', 'Categórica', m.type === 'cat')}${opt('id', 'Identificador', m.type === 'id')}${opt('skip', 'Excluir', m.type === 'skip')}</select></td><td>${isCat ? `<select data-k="ref">${m.levels.map(l => opt(l, l, l === m.ref)).join('')}</select>` : ''}</td><td>${isCat ? `<select data-k="event">${m.levels.map(l => opt(l, l, l === m.event)).join('')}</select>` : ''}</td><td class="num">${m.type === 'skip' ? '' : `${miss} (${pct(miss / state.raw.length)})`}</td><td class="sub" style="white-space:normal">${esc(s)}</td></tr>`;
  }).join('')}</tbody>`;
  $$('select', T).forEach(s => s.addEventListener('change', () => { const v = s.closest('tr').dataset.v, k = s.dataset.k, m = state.meta[v]; if (k === 'type' && s.value === 'num' && !m.allNum) { alert(`«${v}» contiene valores no numéricos.`); s.value = m.type; return; } m[k] = s.value; rebuild(); if (k === 'type') renderVarTable(); state.draft = null; }));
}

// ---------------- 3 · modelo ----------------
const FORMS = [['lin', 'Lineal'], ['log', 'Logaritmo'], ['rcs3', 'Spline 3 nudos'], ['rcs4', 'Spline 4 nudos'], ['rcs5', 'Spline 5 nudos']];
const readVars = () => Object.fromEntries($$('#m-vars input:checked').map(i => [i.dataset.pv, $(`[data-pf="${CSS.escape(i.dataset.pv)}"]`)?.value || 'lin']));
function captureDraft() { if (!$('#m-vars').children.length) return state.draft || {}; return { type: $('#m-type').value, y: $('#m-y').value, time: $('#m-time').value, status: $('#m-status').value, horizon: +$('#m-horizon').value, vars: readVars() }; }
function renderModelForm() {
  if (!state.ds) return; const d = state.draft || {}, type = d.type || $('#m-type').value; $('#m-type').value = type;
  const cats = usable('cat'), nums = usable('num'), all = usable('cat', 'num'), cls = { binary: 'mb', continuous: 'mc', survival: 'ms' }[type];
  $('#m-y').innerHTML = (type === 'continuous' ? nums : cats).map(v => opt(v, v, v === d.y)).join('');
  $('#m-time').innerHTML = nums.map(v => opt(v, v, v === d.time)).join(''); $('#m-status').innerHTML = all.map(v => opt(v, v, v === d.status)).join('');
  syncEvents();
  $$('#tab-modelo label.mb, #tab-modelo label.mc, #tab-modelo label.ms').forEach(e => (e.hidden = !e.classList.contains(cls)));
  $('#m-lasso').hidden = type === 'survival'; if (type === 'survival' && $('#m-pen').value === 'lasso') $('#m-pen').value = 'ridge';
  if (d.horizon) $('#m-horizon').value = d.horizon;
  const skip = new Set(type === 'survival' ? [$('#m-time').value, $('#m-status').value] : [$('#m-y').value]);
  $('#m-vars').innerHTML = all.filter(v => !skip.has(v)).map(v => {
    const m = state.meta[v], sel = d.vars ? d.vars[v] : null;
    return `<label class="pv"><input type="checkbox" data-pv="${esc(v)}"${sel ? ' checked' : ''}> ${esc(v)}${m.type === 'cat' ? ' <span class="badge cat">categ.</span>' : ''} ${m.type === 'num' ? `<select data-pf="${esc(v)}">${FORMS.map(([k, l]) => opt(k, l, k === (sel || 'lin'))).join('')}</select>` : ''}</label>`;
  }).join('');
  $$('#m-vars input, #m-vars select').forEach(e => e.addEventListener('change', modelCheck)); modelCheck();
}
function syncEvents() {
  const lv = v => (state.meta[v]?.levels || []), y = $('#m-y').value, st = $('#m-status').value;
  $('#m-event').innerHTML = lv(y).map(l => opt(l, l, l === state.meta[y]?.event)).join(''); $('#m-sevent').innerHTML = lv(st).map(l => opt(l, l, l === state.meta[st]?.event)).join('');
}
$('#m-type').addEventListener('change', () => { state.draft = { type: $('#m-type').value }; renderModelForm(); });
['#m-y', '#m-time', '#m-status'].forEach(s => $(s).addEventListener('change', () => { state.draft = captureDraft(); renderModelForm(); }));
$('#m-all').addEventListener('click', () => { $$('#m-vars input').forEach(i => (i.checked = true)); modelCheck(); });
$('#m-none').addEventListener('click', () => { $$('#m-vars input').forEach(i => (i.checked = false)); modelCheck(); });
$('#m-pen').addEventListener('change', modelCheck);

function readSpec() {
  const type = $('#m-type').value, vars = Object.entries(readVars()).map(([name, form]) => ({ name, form }));
  const spec = { type, vars, penalty: $('#m-pen').value, lambdaRule: $('#m-rule').value, B: +$('#m-B').value, seed: +$('#m-seed').value || 1, retune: $('#m-retune').checked };
  if (type === 'survival') Object.assign(spec, { time: $('#m-time').value, status: $('#m-status').value, event: $('#m-sevent').value, horizon: +$('#m-horizon').value });
  else Object.assign(spec, { y: $('#m-y').value, event: type === 'binary' ? $('#m-event').value : null });
  return spec;
}
const applyForms = spec => materialize(spec, state.ds, state.meta);
function countParams(spec, ds, meta) { try { return columnNames(designInfo(ds, meta, spec, Array.from({ length: ds.n }, (_, i) => i))).length; } catch { return NaN; } }
function modelCheck() {
  const box = $('#m-check'); if (!state.ds) return;
  const s = readSpec(); if (!s.vars.length) { box.innerHTML = '<p>Marque al menos un predictor.</p>'; return; }
  try {
    const a = applyForms(s), rows = completeFor(a.ds, a.spec), p = countParams(a.spec, a.ds, a.meta), N = rows.length, drop = state.ds.n - N;
    let ev = NaN; if (s.type === 'binary') ev = rows.filter(i => String(a.ds.cols[s.y][i]) === String(s.event)).length; else if (s.type === 'survival') ev = rows.filter(i => String(a.ds.cols[s.status][i]) === String(s.event)).length;
    const ss = state.ss.dev?.result, warn = [];
    if (isFinite(ev) && ev / p < 10) warn.push(`Hay ${f2(ev / p, 1)} eventos por parámetro candidato (EPP): sobreajuste probable; use penalización o reduzca los parámetros.`);
    if (drop / state.ds.n > 0.05) warn.push(`El análisis de casos completos descarta ${drop} de ${state.ds.n} pacientes (${pct(drop / state.ds.n)}). Considere imputar (EPI-DATA, paso 5) y cargar la base imputada.`);
    if (ss && ss.n > N) warn.push(`Su muestra (${N}) es menor que el mínimo calculado en el paso 1 (${ss.n}).`);
    box.innerHTML = `<p><b>${isFinite(p) ? p : '–'}</b> parámetros candidatos · <b>${N}</b> pacientes con datos completos${isFinite(ev) ? ` · <b>${ev}</b> eventos · EPP = <b>${f2(ev / p, 1)}</b>` : ''}. ${ss ? `Mínimo del paso 1: <b>${ss.n.toLocaleString('es')}</b>.` : 'Calcule el mínimo en el paso 1 con este número de parámetros.'}</p>${warn.map(w => `<div class="warn">${esc(w)}</div>`).join('')}`;
  } catch (e) { box.innerHTML = errBox(e); }
}
$('#m-run').addEventListener('click', async () => {
  const btn = $('#m-run'), spec = readSpec(), prog = $('#m-prog'), msg = $('#m-status');
  try {
    if (!spec.vars.length) throw new Error('Marque al menos un predictor.');
    if (spec.type !== 'survival' && state.meta[spec.y].type === 'cat' && spec.type === 'continuous') throw new Error('El desenlace continuo debe ser cuantitativo.');
    btn.disabled = true; prog.hidden = false; prog.value = 0; msg.textContent = 'Ajustando el modelo…'; await tick();
    const a = applyForms(spec), rows = completeFor(a.ds, a.spec), seed = spec.seed;
    const model = fitPred(a.ds, a.meta, a.spec, rows, { seed });
    state.dsX = a.ds; state.metaX = a.meta; state.specX = a.spec; state.spec = spec;
    const bootOpt = { B: spec.B, seed, horizon: spec.horizon, retune: spec.retune, onProgress: p => { prog.value = p; msg.textContent = `Bootstrap: ${Math.round(p * spec.B)} de ${spec.B} réplicas…`; } };
    state.model = model; state.final = model; state.shrunk = null; state.dev = { app: evaluate(model, a.ds, a.meta, rows, { horizon: spec.horizon }), rows };
    state.boot = spec.B > 0 ? await bootstrapValidate(a.ds, a.meta, a.spec, rows, model, bootOpt) : null;
    msg.textContent = ''; prog.hidden = true; gate(); renderDev(); showTab('desarrollo');
  } catch (e) { msg.innerHTML = ''; $('#m-check').insertAdjacentHTML('beforeend', errBox(e)); prog.hidden = true; console.error(e); }
  btn.disabled = false;
});

// ---------------- inicio ----------------
initSampleSize(); initValidation(); bindFigure();
$('#fig-width').addEventListener('change', () => ($('#fig-mm').disabled = $('#fig-width').value !== 'custom')); $('#fig-mm').disabled = true;
gate();
window.__epi = { state };
