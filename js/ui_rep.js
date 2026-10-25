// Pestaña 7: texto de métodos, tablas, ecuación, calculadora de riesgo y lista TRIPOD+AI.
import { $, $$, esc, f2, fp, pct, ci, table, howto, card, state, download } from './state.js?v=2026.10.25-1';
import { predictLPs, riskFromLP, materialize, evaluate } from './predmodel.js?v=2026.10.25-1';
import { coefTable, equationText, exportModel, modelName } from './ui_dev.js?v=2026.10.25-1';
import { wordDoc, matrixHTML, matrixCSV } from './report.js?v=2026.10.25-1';

const PEN = { none: 'máxima verosimilitud sin penalización', ridge: 'regresión ridge', lasso: 'regresión lasso' };
const PEN_EN = { none: 'maximum likelihood without penalisation', ridge: 'ridge regression', lasso: 'lasso regression' };
const FORM = { lin: 'lineal', log: 'logarítmica (ln)', rcs3: 'spline cúbico restringido con 3 nudos', rcs4: 'spline cúbico restringido con 4 nudos', rcs5: 'spline cúbico restringido con 5 nudos' };
const FORM_EN = { lin: 'linear', log: 'natural-log transformed', rcs3: 'restricted cubic spline (3 knots)', rcs4: 'restricted cubic spline (4 knots)', rcs5: 'restricted cubic spline (5 knots)' };

export function methodsText(lang = 'es') {
  const m = state.model, f = state.final, sp = state.spec, es = lang === 'es', ss = state.ss.dev, b = state.boot, N = state.raw.length, ev = m.events;
  const outcome = sp.type === 'survival' ? `${sp.status} (${es ? 'evento' : 'event'}: ${sp.event}; ${es ? 'tiempo' : 'time'}: ${sp.time}; ${es ? 'horizonte de predicción' : 'prediction horizon'}: ${sp.horizon})` : sp.type === 'binary' ? `${sp.y} (${es ? 'evento' : 'event'}: ${sp.event})` : sp.y;
  const preds = sp.vars.map(v => `${v.name} (${(es ? FORM : FORM_EN)[v.form] || v.form})`).join(', ');
  const missing = N - m.n, mod = { binary: es ? 'regresión logística' : 'logistic regression', survival: es ? 'regresión de Cox' : 'Cox proportional hazards regression', continuous: es ? 'regresión lineal' : 'linear regression' }[m.type];
  const meas = { binary: es ? 'el estadístico C, la pendiente de calibración, la calibración en general (CITL), la puntuación de Brier y el R² de Cox-Snell' : 'the C statistic, calibration slope, calibration-in-the-large, Brier score and Cox-Snell R²', survival: es ? 'el índice C de Harrell, la pendiente de calibración y el R² de Cox-Snell, además de la calibración del riesgo a ' + sp.horizon : 'Harrell\'s C index, the calibration slope and Cox-Snell R², together with the calibration of the risk at ' + sp.horizon, continuous: es ? 'el R², la pendiente de calibración y el RMSE' : 'R², calibration slope and RMSE' }[m.type];
  const val = state.valEv, dca = !!$('#u-out')?.innerHTML;
  const L = [];
  if (es) {
    L.push(`Se desarrolló un modelo de predicción ${modelName(m)} para ${outcome}, siguiendo las recomendaciones TRIPOD+AI. La base de desarrollo (${esc(state.fname)}) incluyó ${N} pacientes${ev != null ? `; tras excluir ${missing} con datos incompletos (análisis de casos completos), el modelo se ajustó a ${m.n} pacientes con ${ev} eventos` : `; el modelo se ajustó a ${m.n} pacientes con datos completos`}.`);
    L.push(`Los predictores candidatos fueron ${preds}, definidos a priori por criterio clínico (${m.cols.length} parámetros). ${ss ? `El tamaño de muestra mínimo se calculó con los criterios de Riley et al. (${ss.result.n.toLocaleString('es')} pacientes${ss.result.events ? `, ${f2(ss.result.events, 0)} eventos` : ''}).` : ''}`);
    L.push(`El modelo se estimó mediante ${mod} con ${PEN[m.penalty] || PEN.none}${m.pen?.lambda ? ` (λ = ${f2(m.pen.lambda, 4)} elegido por validación cruzada de 10 pliegues, regla ${m.pen.rule})` : ''}${state.shrunk ? `, y se aplicó una contracción uniforme (factor ${f2(b.shrinkage, 3)} estimado por bootstrap)` : ''}.`);
    L.push(b ? `La validación interna se hizo con bootstrap (${b.B} réplicas), repitiendo en cada una todo el proceso de modelado, para estimar el optimismo y corregir ${meas}.` : `No se realizó validación interna; el rendimiento informado es aparente.`);
    if (val) L.push(`El modelo se validó externamente en ${val.perf.n} pacientes (${esc(state.valLabel)}) sin reajustarlo; se informan ${meas} con sus intervalos de confianza del 95 %.`);
    if (dca) L.push('La utilidad clínica se evaluó con el análisis de curva de decisión (beneficio neto) en un rango de umbrales de riesgo.');
    L.push('El análisis se realizó con EPI-PREDIC (EPI-LAB; Fernández-Guzmán D.), cuyos motores fueron validados frente a statsmodels, scikit-learn, lifelines y pmsampsize.');
  } else {
    L.push(`A ${modelName(m) === 'logístico' ? 'logistic' : modelName(m) === 'de Cox' ? 'Cox' : 'linear'} prediction model was developed for ${outcome}, following the TRIPOD+AI recommendations. The development dataset (${esc(state.fname)}) included ${N} participants${ev != null ? `; after excluding ${missing} with incomplete data (complete-case analysis), the model was fitted in ${m.n} participants with ${ev} events` : `; the model was fitted in ${m.n} participants with complete data`}.`);
    L.push(`Candidate predictors were ${preds}, pre-specified on clinical grounds (${m.cols.length} parameters). ${ss ? `The minimum sample size was calculated with the Riley et al. criteria (${ss.result.n.toLocaleString('en')} participants${ss.result.events ? `, ${f2(ss.result.events, 0)} events` : ''}).` : ''}`);
    L.push(`The model was estimated by ${mod} using ${PEN_EN[m.penalty] || PEN_EN.none}${m.pen?.lambda ? ` (λ = ${f2(m.pen.lambda, 4)} chosen by 10-fold cross-validation, ${m.pen.rule} rule)` : ''}${state.shrunk ? `, and uniform shrinkage (factor ${f2(b.shrinkage, 3)} estimated by bootstrap) was applied` : ''}.`);
    L.push(b ? `Internal validation used bootstrapping (${b.B} resamples), repeating the entire modelling process in each, to estimate optimism and correct ${meas}.` : 'No internal validation was performed; reported performance is apparent.');
    if (val) L.push(`The model was externally validated in ${val.perf.n} participants (${esc(state.valLabel)}) without refitting; ${meas} are reported with 95% confidence intervals.`);
    if (dca) L.push('Clinical utility was assessed with decision curve analysis (net benefit) over a range of risk thresholds.');
    L.push('Analyses were run in EPI-PREDIC (EPI-LAB; Fernández-Guzmán D.), whose engines were validated against statsmodels, scikit-learn, lifelines and pmsampsize.');
  }
  return L.join('\n\n');
}

const TRIPOD = [
  ['Título y resumen', [['1', 'Título: identifica el estudio como desarrollo/validación de un modelo de predicción, la población diana y el desenlace.'], ['2', 'Resumen estructurado (TRIPOD+AI para resúmenes).']]],
  ['Introducción', [['3a', 'Contexto médico y justificación; modelos existentes y sus limitaciones.'], ['3b', 'Brechas de conocimiento (desarrollo, validación o actualización).'], ['4', 'Objetivos: población, desenlace, momento de uso previsto y tipo de estudio.']]],
  ['Métodos', [['5a', 'Fuente de los datos y diseño del estudio (cohorte, ensayo, registro).', 'auto'], ['5b', 'Fechas: reclutamiento, seguimiento y extracción de los datos.'], ['6a', 'Entorno y criterios de elegibilidad de los participantes.'], ['7', 'Preparación de los datos (limpieza, transformaciones, partición).', 'auto'], ['8a', 'Definición del desenlace, horizonte temporal y cómo se evaluó.', 'auto'], ['8b', 'Evaluación del desenlace ciega a los predictores.'], ['9a', 'Predictores: definición, momento de medición y forma funcional.', 'auto'], ['10', 'Tamaño de muestra: justificación (p. ej., Riley et al.).', 'ss'], ['11', 'Datos perdidos: cantidad y cómo se manejaron.', 'auto'], ['12a', 'Método de desarrollo del modelo y selección de predictores.', 'auto'], ['12b', 'Manejo de la no linealidad e interacciones.', 'auto'], ['12c', 'Estrategia de penalización/contracción.', 'auto'], ['12d', 'Validación interna (bootstrap, validación cruzada) y corrección del optimismo.', 'boot'], ['12e', 'Medidas de rendimiento: discriminación, calibración y utilidad clínica.', 'auto'], ['12f', 'Validación externa: datos y métodos.', 'val'], ['13', 'Desbalance de clases y cómo se abordó.'], ['14', 'Equidad: rendimiento por subgrupos relevantes.'], ['15', 'Resultado del modelo: ecuación completa o formato de uso.', 'auto'], ['17', 'Aprobación ética y consentimiento.'], ['18', 'Ciencia abierta: protocolo, código, datos, modelo accesible.'], ['19', 'Participación de pacientes y público.']]],
  ['Resultados', [['20a', 'Flujo de participantes (diagrama) y eventos.', 'auto'], ['20b', 'Características de los participantes (tabla 1); comparación entre desarrollo y validación.'], ['21', 'Especificación del modelo: coeficientes con su incertidumbre, intercepto/riesgo basal.', 'auto'], ['22', 'Rendimiento con IC: discriminación, calibración (gráfico) y utilidad.', 'boot'], ['23', 'Actualización del modelo (recalibración), si procede.', 'rc']]],
  ['Discusión', [['24', 'Interpretación: resultados principales, comparación con modelos previos.'], ['25', 'Limitaciones: representatividad, datos perdidos, sobreajuste, generalización.'], ['26', 'Usabilidad: dónde y cómo se usaría; implicaciones clínicas.']]],
];
function autoStatus(kind) { return kind === 'auto' || (kind === 'ss' && state.ss.dev) || (kind === 'boot' && state.boot) || (kind === 'val' && state.valEv) || (kind === 'rc' && state.recalM); }

// ---------------------------------------------------------------- calculadora de riesgo
function calcHTML() {
  const sp = state.spec, m = state.final;
  return `<div class="form">${sp.vars.map(v => { const mm = state.meta[v.name]; return mm.type === 'cat' ? `<label>${esc(v.name)} <select data-cv="${esc(v.name)}">${mm.levels.map(l => `<option>${esc(l)}</option>`).join('')}</select></label>` : `<label>${esc(v.name)} <input data-cv="${esc(v.name)}" type="number" step="any" value="${f2(median(state.ds.cols[v.name]), 2)}"></label>`; }).join('')}</div><div class="kpis"><div class="kpi"><b id="c-risk">–</b><span>${m.type === 'survival' ? `riesgo a ${sp.horizon}` : m.type === 'binary' ? 'riesgo predicho' : 'valor predicho'}</span></div></div>`;
}
const median = a => { const v = Array.from(a).filter(isFinite).sort((x, y) => x - y); return v[Math.floor(v.length / 2)]; };
function calcRun() {
  try {
    const sp = state.spec, cols = {}; for (const el of $$('[data-cv]')) { const nm = el.dataset.cv, mm = state.meta[nm]; cols[nm] = mm.type === 'cat' ? [el.value] : Float64Array.of(parseFloat(el.value)); }
    const mt = materialize(sp, { n: 1, cols }, state.meta), pr = predictLPs(state.final, mt.ds, mt.meta, [0]);
    if (!pr.lp.length) { $('#c-risk').textContent = '–'; return; }
    const r = riskFromLP(state.final, pr.lp, sp.horizon)[0]; $('#c-risk').textContent = state.final.type === 'continuous' ? f2(r, 2) : pct(r, 1);
  } catch { $('#c-risk').textContent = '–'; }
}

export function renderReport() {
  const out = $('#rep-out'); if (!state.final) { out.innerHTML = ''; return; }
  const m = state.final, ct = coefTable(m), b = state.boot;
  const perfRows = b ? b.rows.map(r => ({ c: [r.key, f2(r.apparent, 3), f2(r.optimism, 3), f2(r.corrected, 3)] })) : [];
  const checklist = TRIPOD.map(([sec, items]) => `<h3>${esc(sec)}</h3><table class="data chk"><tbody>${items.map(([n, t, kind]) => { const auto = autoStatus(kind); return `<tr><td><input type="checkbox" data-tr="${n}"${auto ? ' checked' : ''} title="${auto ? 'Cubierto por el reporte generado' : 'Marcar cuando esté redactado'}"></td><td><b>${n}</b></td><td>${esc(t)}${auto ? ' <span class="badge">generado</span>' : ''}</td></tr>`; }).join('')}</tbody></table>`).join('');
  out.innerHTML = [
    card('Texto de métodos', `<div class="row gap"><button class="ghost" data-lang="es">Español</button><button class="ghost" data-lang="en">English</button><span class="grow"></span><button class="ghost" id="r-copy">Copiar</button></div><textarea id="methods" rows="14"></textarea>`),
    card('Tablas', `<div class="row gap wrap"><button class="ghost" id="r-doc">Word (.doc)</button><button class="ghost" id="r-csv">CSV</button></div><h3>Coeficientes del modelo</h3>${table(ct.head, ct.rows)}${b ? `<h3>Rendimiento (aparente, optimismo, corregido)</h3>${table(['Medida', 'Aparente', 'Optimismo', 'Corregido'], perfRows.map(r => r.c))}` : ''}`),
    card('Ecuación y calculadora de riesgo', `<pre class="eq">${esc(equationText(m, state.spec.horizon))}</pre><div class="row gap"><button class="ghost" id="r-eq">Copiar ecuación</button><button class="ghost" id="r-json">Descargar modelo (.json)</button></div><h3>Calculadora de riesgo individual</h3>${calcHTML()}`),
    card('Lista TRIPOD+AI', `<p class="sub">Los elementos marcados «generado» los cubre el reporte; complete los demás en el manuscrito. Referencia: Collins GS et al. <i>BMJ</i> 2024;385:e078378 (TRIPOD+AI). Para la comparación con otras guías véase la serie de Riley et al. (<i>BMJ</i> 2024;384:e074819, e074820, e074821).</p>${checklist}<p class="sub" id="tr-n"></p>`),
  ].join('');
  let lang = 'es'; const set = () => { $('#methods').value = methodsText(lang); }; set();
  $$('[data-lang]').forEach(bn => bn.addEventListener('click', () => { lang = bn.dataset.lang; set(); }));
  $('#r-copy').addEventListener('click', () => navigator.clipboard?.writeText($('#methods').value));
  $('#r-eq').addEventListener('click', () => navigator.clipboard?.writeText(equationText(m, state.spec.horizon)));
  $('#r-json').addEventListener('click', () => download('modelo_prediccion.json', JSON.stringify(exportModel(m), null, 1), 'application/json'));
  const mat = { head: ct.head, body: ct.rows.map(r => ({ c: r.map(x => String(x).replace(/<[^>]+>/g, '').replace(/&nbsp;/g, ' ')) })) };
  $('#r-doc').addEventListener('click', () => download('modelo_prediccion.doc', wordDoc('Modelo de predicción', matrixHTML(mat) + (b ? matrixHTML({ head: ['Medida', 'Aparente', 'Optimismo', 'Corregido'], body: perfRows }) : '')), 'application/msword'));
  $('#r-csv').addEventListener('click', () => download('modelo_prediccion.csv', matrixCSV(mat), 'text/csv'));
  $$('[data-cv]').forEach(el => el.addEventListener('input', calcRun)); calcRun();
  const cnt = () => { const all = $$('[data-tr]'), on = all.filter(x => x.checked).length; $('#tr-n').textContent = `${on} de ${all.length} elementos cubiertos.`; }; $$('[data-tr]').forEach(x => x.addEventListener('change', cnt)); cnt();
}
