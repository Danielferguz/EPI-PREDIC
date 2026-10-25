// Pestaña 7: texto de métodos, tablas, ecuación, calculadora de riesgo y lista TRIPOD+AI.
import { $, $$, esc, f2, fp, pct, ci, table, howto, card, state, download } from './state.js?v=2026.10.25-7';
import { predictLPs, riskFromLP, materialize, evaluate } from './predmodel.js?v=2026.10.25-7';
import { coefTable, equationText, exportModel, modelName } from './ui_dev.js?v=2026.10.25-7';
import { wordDoc, matrixHTML, matrixCSV } from './report.js?v=2026.10.25-7';
import { levelsOf } from './data.js?v=2026.10.25-7';
import { pythonScript } from './pycode.js?v=2026.10.25-7';
import { calcPageHTML } from './calcpage.js?v=2026.10.25-7';
import { buildManuscript } from './manuscript.js?v=2026.10.25-7';

const PEN = { none: 'máxima verosimilitud sin penalización', ridge: 'regresión ridge', lasso: 'regresión lasso' };
const PEN_EN = { none: 'maximum likelihood without penalisation', ridge: 'ridge regression', lasso: 'lasso regression' };
const FORM = { lin: 'lineal', log: 'logarítmica (ln)', rcs3: 'spline cúbico restringido con 3 nudos', rcs4: 'spline cúbico restringido con 4 nudos', rcs5: 'spline cúbico restringido con 5 nudos' };
const FORM_EN = { lin: 'linear', log: 'natural-log transformed', rcs3: 'restricted cubic spline (3 knots)', rcs4: 'restricted cubic spline (4 knots)', rcs5: 'restricted cubic spline (5 knots)' };

export function methodsText(lang = 'es') {
  const m = state.model, sp = state.spec, es = lang === 'es', ss = state.ss.dev, b = state.boot, N = state.raw.length, ev = m.events, type = m.type, sh = state.simpApplied;
  const outcome = type === 'survival' ? `${sp.status} (${es ? 'evento' : 'event'}: ${sp.event}; ${es ? 'tiempo' : 'time'}: ${sp.time}; ${es ? 'horizonte de predicción' : 'prediction horizon'}: ${sp.horizon})` : type === 'binary' ? `${sp.y} (${es ? 'evento' : 'event'}: ${sp.event})` : sp.y;
  const preds = sp.vars.map(v => `${v.name} (${(es ? FORM : FORM_EN)[v.form] || v.form})`).join(', '), missing = N - m.n, E = type === 'binary' ? 'OR' : type === 'survival' ? 'HR' : 'β';
  const mod = { binary: es ? 'regresión logística (statsmodels.Logit)' : 'logistic regression (statsmodels.Logit)', survival: es ? 'regresión de Cox con empates de Efron (lifelines.CoxPHFitter)' : 'Cox proportional hazards regression with Efron ties (lifelines.CoxPHFitter)', continuous: es ? 'regresión lineal (statsmodels.OLS)' : 'linear regression (statsmodels.OLS)' }[type];
  const meas = { binary: es ? 'el estadístico C (sklearn.metrics.roc_auc_score), la pendiente de calibración, la calibración en general (CITL), la puntuación de Brier y el R² de Cox-Snell' : 'the C statistic (sklearn.metrics.roc_auc_score), calibration slope, calibration-in-the-large (CITL), Brier score and Cox-Snell R²', survival: es ? `el índice C de Harrell (lifelines.utils.concordance_index) y la pendiente de calibración, además de la calibración del riesgo a ${sp.horizon}` : `Harrell's C-index (lifelines.utils.concordance_index) and the calibration slope, together with calibration of the risk at ${sp.horizon}`, continuous: es ? 'el R², la pendiente de calibración y el RMSE' : 'R², calibration slope and RMSE' }[type];
  const val = state.valEv, dca = !!$('#u-out')?.innerHTML, L = [];
  if (es) {
    L.push(`Se desarrolló un modelo de predicción para ${outcome}, siguiendo TRIPOD+AI. La base incluyó ${N} participantes${ev != null ? `; tras excluir ${missing} con datos incompletos (casos completos), el modelo se ajustó a ${m.n} con ${ev} eventos` : `; el modelo se ajustó a ${m.n} con datos completos`}. Los predictores, definidos a priori por criterio clínico, fueron ${preds} (${m.cols.length} parámetros). ${ss ? `El tamaño de muestra mínimo se calculó con los criterios de Riley et al. (${ss.result.n.toLocaleString('es')} participantes${ss.result.events ? `, ${f2(ss.result.events, 0)} eventos` : ''}).` : ''}`);
    L.push(`Las características basales se describieron en el total y según el desenlace con mediana (rango intercuartílico) o n (%), y se compararon con la U de Mann-Whitney (scipy.stats.mannwhitneyu), la χ² de Pearson o la prueba exacta de Fisher (scipy.stats.chi2_contingency, fisher_exact). La asociación de cada predictor se estimó con regresión cruda y ajustada (${E} con IC 95 %).`);
    L.push(`El modelo se estimó mediante ${mod}${m.penalty === 'ridge' ? ' con penalización ridge (L2)' : m.penalty === 'lasso' ? ' con penalización lasso (L1)' : ''}${m.pen?.lambda ? ` (λ = ${f2(m.pen.lambda, 4)} elegido por validación cruzada de 10 pliegues, scikit-learn)` : ''}; las variables cuantitativas con relación no lineal se modelaron con splines cúbicos restringidos de Harrell${state.shrunk ? `, y se aplicó una contracción uniforme (factor ${f2(b.shrinkage, 3)} estimado por bootstrap)` : ''}.`);
    L.push(b ? `La validación interna se hizo con bootstrap (${b.B} réplicas), repitiendo en cada una todos los pasos de modelado, para corregir el optimismo de ${meas}.` : 'No se realizó validación interna; el rendimiento informado es aparente.');
    if (sh) L.push(`Para simplificar el modelo, los predictores se ordenaron por su χ² de Wald parcial y se añadieron de uno en uno; se retuvo el modelo más pequeño cuyo rendimiento corregido por bootstrap no disminuyó más de ${sh.tol} (${sh.names.length} de ${sh.from} predictores).`);
    if (val) L.push(`El modelo se validó externamente en ${val.perf.n} participantes (${esc(state.valLabel)}) sin reajustarlo; se informan ${meas} con intervalos de confianza del 95 %.`);
    if (dca) L.push('La utilidad clínica se evaluó con el análisis de curva de decisión (beneficio neto) en un rango de umbrales de riesgo.');
    L.push('Los métodos estadísticos corresponden a las implementaciones de Python indicadas (numpy, pandas, SciPy, statsmodels, scikit-learn' + (type === 'survival' ? ', lifelines' : '') + '); se proporciona un script equivalente como material suplementario.');
  } else {
    L.push(`A prediction model for ${outcome} was developed following TRIPOD+AI. The dataset included ${N} participants${ev != null ? `; after excluding ${missing} with incomplete data (complete-case analysis), the model was fitted in ${m.n} participants with ${ev} events` : `; the model was fitted in ${m.n} participants with complete data`}. Predictors, pre-specified on clinical grounds, were ${preds} (${m.cols.length} parameters). ${ss ? `The minimum sample size was calculated with the Riley et al. criteria (${ss.result.n.toLocaleString('en')} participants${ss.result.events ? `, ${f2(ss.result.events, 0)} events` : ''}).` : ''}`);
    L.push(`Baseline characteristics were described overall and by outcome as median (interquartile range) or n (%), and compared with the Mann–Whitney U test (scipy.stats.mannwhitneyu), Pearson's χ² or Fisher's exact test (scipy.stats.chi2_contingency, fisher_exact). The association of each predictor was estimated with crude and adjusted regression (${E} with 95% CI).`);
    L.push(`The model was estimated by ${mod}${m.penalty === 'ridge' ? ' with ridge (L2) penalisation' : m.penalty === 'lasso' ? ' with lasso (L1) penalisation' : ''}${m.pen?.lambda ? ` (λ = ${f2(m.pen.lambda, 4)} chosen by 10-fold cross-validation, scikit-learn)` : ''}; continuous predictors with non-linear relationships were modelled with Harrell's restricted cubic splines${state.shrunk ? `, and uniform shrinkage (factor ${f2(b.shrinkage, 3)} estimated by bootstrap) was applied` : ''}.`);
    L.push(b ? `Internal validation used bootstrapping (${b.B} resamples), repeating every modelling step in each, to correct the optimism of ${meas}.` : 'No internal validation was performed; reported performance is apparent.');
    if (sh) L.push(`To simplify the model, predictors were ranked by their partial Wald χ² and added one at a time; the smallest model whose bootstrap-corrected performance did not decrease by more than ${sh.tol} was retained (${sh.names.length} of ${sh.from} predictors).`);
    if (val) L.push(`The model was externally validated in ${val.perf.n} participants (${esc(state.valLabel)}) without refitting; ${meas} are reported with 95% confidence intervals.`);
    if (dca) L.push('Clinical utility was assessed with decision curve analysis (net benefit) over a range of risk thresholds.');
    L.push('The statistical methods correspond to the Python implementations named above (numpy, pandas, SciPy, statsmodels, scikit-learn' + (type === 'survival' ? ', lifelines' : '') + '); an equivalent script is provided as supplementary material.');
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
// configuración del script Python (sin datos de pacientes)
function pyConfig() {
  const m = state.final, sp = state.spec, S = state.simpApplied, all = S?.all || sp.vars.map(v => ({ name: v.name, form: v.form })), spec = v => { const mm = state.meta[v.name]; return mm.type === 'cat' ? { name: v.name, type: 'cat', form: 'lin', levels: levelsOf(mm) } : { name: v.name, type: 'num', form: v.form }; };
  return { data_file: `${state.fname}.csv`, na_values: ['', 'NA', 'na', 'NaN'], type: m.type, y: sp.y, event: sp.event, time: sp.time, status: sp.status, sevent: sp.event, horizon: sp.horizon, penalty: sp.penalty, B: sp.B, seed: sp.seed,
    predictors: all.map(spec), predictors_raw: all.map(v => { const mm = state.meta[v.name]; return mm.type === 'cat' ? { name: v.name, type: 'cat', levels: levelsOf(mm) } : { name: v.name, type: 'num' }; }),
    simplify: { run: false, B: 100, tol: S?.tol ?? 0.01, selected: S ? S.names : null }, meta: { n: m.n, events: m.events, date: new Date().toISOString().slice(0, 10) } };
}
function calcInfo() {
  const sp = state.spec, m = state.final, rg = v => +Number(v).toPrecision(3), vars = sp.vars.map(v => { const mm = state.meta[v.name], col = Array.from(state.ds.cols[v.name]);
    if (mm.type === 'cat') return { name: v.name, kind: 'cat', levels: levelsOf(mm) }; const x = col.filter(isFinite).sort((a, b) => a - b); return { name: v.name, kind: 'num', min: rg(x[0]), max: rg(x[x.length - 1]), median: rg(x[Math.floor(x.length / 2)]) }; });
  const b = state.boot, key = { binary: 'auc', survival: 'cindex', continuous: 'r2' }[m.type], cr = b?.rows.find(r => r.key === key);
  return { title: `Calculadora · ${sp.y || sp.status}`, outcome: sp.y || sp.status, vars, n: m.n, events: m.events, horizon: sp.horizon, date: new Date().toISOString().slice(0, 10), perf: cr ? `${key === 'auc' ? 'AUC' : key === 'cindex' ? 'índice C' : 'R²'} ${f2(cr.corrected, 3)}` : '', notes: state.simpApplied ? `Modelo simplificado: ${state.simpApplied.names.length} de ${state.simpApplied.from} predictores.` : '' };
}
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
    card('Código Python y calculadora descargable', `<p class="sub">El script replica el análisis (Tabla 1, regresión cruda y ajustada, modelo, bootstrap, simplificación, calibración por grupos, beneficio neto y validación externa) con statsmodels, scikit-learn y lifelines, <b>sin código de figuras</b>. La calculadora es una página HTML autónoma, sin conexión y sin datos de pacientes, para introducir los valores de las variables del modelo y estimar el riesgo; es una herramienta de investigación sin validación externa.</p><div class="row gap wrap"><button class="ghost" id="r-py">Código Python (.py)</button><button class="ghost" id="r-calc">Calculadora de riesgo (HTML)</button></div>`),
    card('Borrador de manuscrito (Español / English)', `${howto('Un borrador con título, resumen, objetivo, métodos y resultados (Tabla 1, regresión cruda y ajustada, coeficientes, rendimiento corregido por bootstrap) y figuras, redactado con los números de su análisis.', 'Tener un punto de partida con estructura de revista (TRIPOD+AI) y la estadística descrita con las funciones de Python equivalentes.', 'Revise y complete la introducción, discusión y referencias. Si los datos son reales, no marque «simulados». Para Word: ábralo en el navegador y copie, o abra el .doc descargado.')}<div class="form"><label>Título (opcional) <input id="ms-title" placeholder="se genera automáticamente"></label><label>Población (para el título) <input id="ms-pop" placeholder="p. ej., adults with chronic kidney disease"></label><label>Nombre del desenlace <input id="ms-out" placeholder="${esc(state.spec.y || state.spec.status)}"></label><label class="inline"><input type="checkbox" id="ms-sim"${state.isExample ? ' checked' : ''}> Los datos son simulados (ejemplo)</label></div><div class="row gap wrap"><button class="primary" id="ms-es">Generar en español</button><button class="primary" id="ms-en">Generate in English</button><button class="ghost" id="ms-dl" disabled>Descargar HTML</button><button class="ghost" id="ms-doc" disabled>Descargar .doc</button><span id="ms-msg" class="muted"></span></div><iframe id="ms-view" style="width:100%;height:520px;border:1px solid var(--line);border-radius:8px;margin-top:10px" hidden></iframe>`),
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
  $('#r-py').addEventListener('click', () => download('replicar_modelo.py', pythonScript(pyConfig()), 'text/x-python'));
  $('#r-calc').addEventListener('click', () => download('calculadora_riesgo.html', calcPageHTML(exportModel(m), calcInfo()), 'text/html'));
  let ms = null; const gen = async lang => { const msg = $('#ms-msg'); msg.textContent = 'Generando…'; try { state.isExample = $('#ms-sim').checked; ms = await buildManuscript({ lang, title: $('#ms-title').value.trim(), pop: $('#ms-pop').value.trim(), outcome: $('#ms-out').value.trim() }); const fr = $('#ms-view'); fr.hidden = false; fr.srcdoc = ms.html; $('#ms-dl').disabled = $('#ms-doc').disabled = false; msg.textContent = ''; } catch (e) { msg.textContent = e.message; console.error(e); } };
  $('#ms-es').addEventListener('click', () => gen('es')); $('#ms-en').addEventListener('click', () => gen('en'));
  $('#ms-dl').addEventListener('click', () => ms && download('manuscrito.html', ms.html, 'text/html')); $('#ms-doc').addEventListener('click', () => ms && download('manuscrito.doc', ms.html, 'application/msword'));
  const cnt = () => { const all = $$('[data-tr]'), on = all.filter(x => x.checked).length; $('#tr-n').textContent = `${on} de ${all.length} elementos cubiertos.`; }; $$('[data-tr]').forEach(x => x.addEventListener('change', cnt)); cnt();
}
