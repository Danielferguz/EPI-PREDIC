// Tarjetas adicionales de la pestaña «Desarrollo»: Tabla 1, regresión cruda y ajustada, gráficos de rendimiento y simplificación del modelo.
import { $, $$, esc, f2, fp, pct, table, howto, card, state, errBox, download } from './state.js?v=2026.10.25-6';
import { table1, table1HTML, wordDoc, matrixHTML, matrixCSV } from './report.js?v=2026.10.25-6';
import { groupColumn, crudeAdjusted, uniAdjTable } from './analysis.js?v=2026.10.25-6';
import { calibGroups, calibSurvival } from './predmodel.js?v=2026.10.25-6';
import { riskDistPlot, thresholdPlot, decilePlot, importancePlot, pathPlot } from './pplots.js?v=2026.10.25-6';
import { simplifyPath } from './simplify.js?v=2026.10.25-6';
import { figWrap, bindFigs } from './figs.js?v=2026.10.25-6';

const specOf = () => { const s = state.spec; return { type: s.type, y: s.y, event: s.event, time: s.time, status: s.status, sevent: s.event }; };
const names = () => (state.simpApplied?.all || state.spec.vars).map(v => v.name); // Tabla 1: todos los candidatos, aunque el modelo final se haya simplificado
export const t1Table = (lang = 'es', nonparam = true) => { const g = groupColumn(state.ds, state.meta, specOf()); return { ...table1(g.ds, g.meta, [...names(), ...(state.spec.type === 'continuous' ? [] : [])], g.group, { lang, nonparam }), group: g.label }; };
export const uaTable = (lang = 'es') => { const sp = specOf(), R = crudeAdjusted(state.dsX, state.metaX, { type: sp.type, vars: state.specX.vars.map(v => v.name), y: sp.y, event: sp.event, time: sp.time, status: sp.status, sevent: sp.sevent, perSD: !!$('#ua-sd')?.checked }); return { R, T: uniAdjTable(R, lang) }; };
const tHTML = T => `<div class="tablewrap"><table class="data t1"><thead><tr>${T.head.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${T.rows.map((r, i) => `<tr>${r.map((c, k) => `<td${T.bold[i] && k === 0 ? ' class="b"' : ''}>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;

export function moreCards() {
  const sp = state.spec, type = sp.type, bin = type === 'binary';
  const t1 = `<label class="inline"><input type="checkbox" id="t1-np" checked> Mediana (RIC) y prueba no paramétrica para todas las cuantitativas</label> <label class="inline">Idioma de la tabla <select id="t1-lang"><option value="es">Español</option><option value="en">English</option></select></label>`;
  return [
    card('Tabla 1 · Características de los participantes', `${howto('Cada variable en el total y según ' + (type === 'continuous' ? 'terciles del desenlace' : 'el desenlace') + ', con el valor p de la comparación entre grupos (U de Mann-Whitney o t de Welch; χ² o exacta de Fisher; Kruskal-Wallis/ANOVA si hay más de 2 grupos).', 'Describir la muestra de desarrollo y ver en qué se diferencian quienes tienen y no tienen el desenlace; es la «Tabla 1» que exigen las revistas (TRIPOD+AI ítem 20b).', 'Los valores p describen diferencias entre grupos; <b>no</b> son un criterio para elegir predictores ni miden su utilidad predictiva. Con muestras grandes, diferencias triviales dan p pequeño. Las cuantitativas se resumen con mediana (RIC) cuando no son normales.')}<div class="row gap wrap">${t1}<button class="ghost" id="t1-doc">Word (.doc)</button><button class="ghost" id="t1-csv">CSV</button></div><div id="t1-out"></div>`),
    card('Regresión por predictor · cruda y ajustada', `${howto('Para cada predictor, el efecto estimado solo (cruda) y junto con todos los demás (ajustada): ' + (bin ? 'razón de odds (OR)' : type === 'survival' ? 'razón de riesgos (HR)' : 'coeficiente β') + ' con IC 95 % y valor p. Las cuantitativas entran de forma lineal (por unidad) y las categóricas contra su categoría de referencia.', 'Mostrar qué variables se asocian con el desenlace y cuánto cambian al ajustar por las demás (confusión/colinealidad entre predictores), como pide la sección de resultados de un artículo.', 'Es un resumen descriptivo con los mismos pacientes (casos completos). Los efectos ajustados <b>no</b> son efectos causales (falacia de la Tabla 2). Si un efecto cambia mucho al ajustar, los predictores comparten información.')}<div class="row gap"><label class="inline"><input type="checkbox" id="ua-sd"> Cuantitativas por 1 DE</label><label class="inline">Idioma <select id="ua-lang"><option value="es">Español</option><option value="en">English</option></select></label><button class="ghost" id="ua-doc">Word (.doc)</button><button class="ghost" id="ua-csv">CSV</button></div><div id="ua-out"></div>`),
    card('Más gráficos para valorar qué tan bien sirve el modelo', `<div id="pf-out"></div>`),
    card('Simplificar el modelo sin perder capacidad de predicción', `${howto('Los predictores se ordenan por su aporte (χ² de Wald parcial − gl) y se añaden de uno en uno; para cada tamaño se estima el rendimiento corregido por bootstrap (el orden se repite en cada réplica, sin optimismo por selección).', 'Encontrar el modelo más pequeño cuyo rendimiento corregido no baja más que una tolerancia respecto al modelo completo: menos variables que recoger, menos sobreajuste y una calculadora más práctica.', 'Mire la curva «corregida»: el punto k marcado es el modelo mínimo dentro de la banda. Si la curva sigue subiendo, no simplifique. Con penalización se usa el mismo λ del modelo completo (aproximación). Valide el modelo simplificado en datos externos.')}<div class="form"><label>Tolerancia de pérdida (${bin ? 'AUC' : type === 'survival' ? 'índice C' : 'R²'}) <input id="sm-tol" type="number" step="0.005" min="0" value="0.01"></label><label>Réplicas bootstrap <input id="sm-B" type="number" min="20" max="500" value="100"></label></div><div class="row gap"><button class="primary" id="sm-run">Calcular</button><progress id="sm-prog" max="1" value="0" hidden></progress><span id="sm-msg" class="muted"></span></div><div id="sm-out"></div>`),
  ].join('');
}

export function bindMore() {
  const doTable = () => { try { const T = t1Table($('#t1-lang').value, $('#t1-np').checked); state.t1 = T; $('#t1-out').innerHTML = `<p class="sub">Grupos: ${esc(T.group)}</p>${table1HTML(T)}`; } catch (e) { $('#t1-out').innerHTML = errBox(e); } };
  $('#t1-np').addEventListener('change', doTable); $('#t1-lang').addEventListener('change', doTable); doTable();
  $('#t1-doc').addEventListener('click', () => { const T = t1Table($('#t1-lang').value, $('#t1-np').checked); download('tabla1.doc', wordDoc('Table 1', matrixHTML({ head: T.head, body: T.rows.map(r => ({ c: r.cells, b: r.bold })) }), T.note), 'application/msword'); });
  $('#t1-csv').addEventListener('click', () => { const T = t1Table($('#t1-lang').value, $('#t1-np').checked); download('tabla1.csv', matrixCSV({ head: T.head, body: T.rows.map(r => ({ c: r.cells })) }), 'text/csv'); });
  const doUA = () => { try { const { R, T } = uaTable($('#ua-lang').value); state.ua = R; $('#ua-out').innerHTML = `<p class="sub">${R.n} pacientes con datos completos${R.events != null ? ` · ${R.events} eventos` : ''}${R.dropped ? ` (${R.dropped} excluidos por datos perdidos)` : ''}${R.adjOk ? '' : ' · el modelo ajustado no convergió (colinealidad o separación)'}.</p>${tHTML(T)}`; } catch (e) { $('#ua-out').innerHTML = errBox(e); } };
  $('#ua-lang').addEventListener('change', doUA); $('#ua-sd').addEventListener('change', doUA); doUA();
  $('#ua-doc').addEventListener('click', () => { const { T } = uaTable($('#ua-lang').value); download('regresion_cruda_ajustada.doc', wordDoc('Table 2', matrixHTML({ head: T.head, body: T.rows.map((r, i) => ({ c: r, b: T.bold[i] })) })), 'application/msword'); });
  $('#ua-csv').addEventListener('click', () => { const { T } = uaTable($('#ua-lang').value); download('regresion_cruda_ajustada.csv', matrixCSV({ head: T.head, body: T.rows.map(r => ({ c: r })) }), 'text/csv'); });
  perfPlots(); bindSimplify();
}

// ---------------------------------------------------------------- gráficos y tabla de clasificación
function perfPlots() {
  const ev = state.dev.app, sp = state.spec, out = $('#pf-out'), type = sp.type; let h = '';
  if (type === 'binary') {
    const p = ev.p, y = ev.y, prev = y.reduce((a, b) => a + b, 0) / y.length, g = calibGroups(p, y, 10);
    const th = [...new Set([prev, 0.05, 0.1, 0.2, 0.3, 0.5].map(v => +v.toFixed(3)))].sort((a, b) => a - b), n = y.length, P = y.reduce((a, b) => a + b, 0), N = n - P;
    const rows = th.map(t => { let tp = 0, fp_ = 0; for (let i = 0; i < n; i++) if (p[i] >= t) { if (y[i]) tp++; else fp_++; } const fn = P - tp, tn = N - fp_, se = tp / P, sp_ = tn / N, ppv = tp / (tp + fp_), npv = tn / (tn + fn); return [`≥ ${(t * 100).toFixed(1)}%${Math.abs(t - prev) < 1e-3 ? ' (prevalencia)' : ''}`, pct(se), pct(sp_), isFinite(ppv) ? pct(ppv) : '–', isFinite(npv) ? pct(npv) : '–', f2(se / (1 - sp_), 2), f2((1 - se) / sp_, 2), `${tp + fp_} (${pct((tp + fp_) / n, 0)})`]; });
    h += `${howto('Cómo se separan los riesgos predichos de quienes tienen y no tienen el evento.', 'Ver de un vistazo la capacidad de discriminar: cuanto menos se solapan las dos distribuciones, mejor.', 'La «pendiente de discriminación» es la diferencia de riesgo medio entre eventos y no eventos. Un modelo útil desplaza la distribución de los eventos hacia riesgos altos.')}${figWrap(riskDistPlot({ pred: p, y }), 'distribucion_riesgos')}`;
    h += `${howto('Sensibilidad, especificidad, valor predictivo positivo (VPP) y negativo (VPN) para cada umbral de riesgo.', 'Elegir un punto de corte según el balance entre falsos negativos y falsos positivos en su contexto clínico.', 'Al subir el umbral baja la sensibilidad y sube la especificidad. El VPP depende de la prevalencia. Un umbral no es «el correcto» en sí mismo: lo fija la decisión clínica (véase la curva de decisión).')}${figWrap(thresholdPlot({ pred: p, y, marks: [prev] }), 'rendimiento_por_umbral')}`;
    h += `${howto('Tabla de clasificación para umbrales seleccionados, con razones de verosimilitud (LR+ y LR−).', 'Cuantificar cuántos pacientes se marcarían como de alto riesgo y con qué aciertos.', 'LR+ &gt; 10 o LR− &lt; 0.1 cambian mucho la probabilidad; entre 0.5 y 2 apenas cambian la decisión. «Positivos» = pacientes por encima del umbral.')}${table(['Umbral de riesgo', 'Sensibilidad', 'Especificidad', 'VPP', 'VPN', 'LR+', 'LR−', 'Positivos'], rows)}`;
    h += `${howto('Riesgo observado (barras, IC 95 %) y riesgo medio predicho (rombos) en 10 grupos de igual tamaño según el riesgo predicho.', 'Comprobar la calibración y ver cuánto sube el riesgo real entre el grupo de menor y el de mayor riesgo.', 'Los rombos deberían caer sobre las barras. Barras que suben de forma monótona indican buena discriminación; rombos sistemáticamente por encima o por debajo indican sobre- o subestimación.')}${figWrap(decilePlot({ groups: g }), 'riesgo_por_grupos')}`;
  } else if (type === 'survival') {
    const g = calibSurvival(ev.p, ev.time, ev.status, sp.horizon, 5);
    h += `${howto(`Riesgo observado (Kaplan-Meier, IC 95 %) y predicho a ${sp.horizon} en 5 grupos de riesgo.`, 'Comprobar la calibración y la separación entre grupos de riesgo.', 'Los rombos deben caer sobre las barras; barras crecientes indican buena discriminación.')}${figWrap(decilePlot({ groups: g, title: `Observed vs predicted risk at ${sp.horizon} by risk group` }), 'riesgo_por_grupos_cox')}`;
  } else h = '<p class="muted">Para un desenlace continuo, el gráfico de calibración «observado vs predicho» de arriba resume el ajuste; añada el R² y el RMSE de la tabla de rendimiento.</p>';
  out.innerHTML = h; bindFigs(out);
}

// ---------------------------------------------------------------- simplificación
function bindSimplify() {
  $('#sm-run').addEventListener('click', async () => {
    const btn = $('#sm-run'), prog = $('#sm-prog'), msg = $('#sm-msg'), out = $('#sm-out'), sp = state.spec, type = sp.type; btn.disabled = true; prog.hidden = false; prog.value = 0; msg.textContent = 'Calculando…'; out.innerHTML = '';
    try {
      if (state.specX.vars.length < 2) throw new Error('Se necesitan al menos 2 predictores.');
      const B = Math.max(20, +$('#sm-B').value || 100), tol = Math.max(0, +$('#sm-tol').value || 0.01), R = await simplifyPath(state.dsX, state.metaX, state.specX, state.dev.rows, state.model, { B, tol, seed: sp.seed || 7, onProgress: p => { prog.value = p; msg.textContent = `Bootstrap ${Math.round(p * B)} de ${B}`; } });
      state.simp = R; const lab = { auc: 'AUC', cindex: 'índice C', r2: 'R²' }[R.key], lab2 = { auc: 'C statistic', cindex: 'C-index', r2: 'R²' }[R.key];
      const orig = nm => state.spec.vars[state.specX.vars.findIndex(v => v.name === nm)]?.name || nm, kept = R.order.slice(0, R.kStar), cut = R.rows[R.kStar - 1], full = R.rows[R.rows.length - 1];
      const imp = importancePlot({ items: R.imp.map(i => ({ label: orig(i.name), value: i.score, keep: R.order.indexOf(i.name) < R.kStar })), xlab: 'Partial Wald χ² − df (full model)' });
      out.innerHTML = `<p><b>Modelo simplificado propuesto: ${R.kStar} de ${R.rows.length} predictores</b> (${kept.map(n => esc(orig(n))).join(', ')}). ${lab} corregido ${f2(cut.corrected, 3)} frente a ${f2(full.corrected, 3)} con todos (diferencia ${f2(full.corrected - cut.corrected, 3)}; tolerancia ${R.tol}). ${R.B} réplicas válidas${R.failed ? ` (${R.failed} descartadas)` : ''}.</p><div class="plots">${figWrap(imp, 'importancia_predictores')}${figWrap(pathPlot({ rows: R.rows, kStar: R.kStar, tol: R.tol, ylab: lab2 }), 'rendimiento_vs_predictores')}</div>${table(['k', 'Predictor añadido', `${lab} aparente`, 'Optimismo', `${lab} corregido`, 'Δ vs completo'], R.rows.map(r => [r.k + (r.k === R.kStar ? ' ◀' : ''), esc(orig(r.added)), f2(r.apparent, 3), f2(r.optimism, 3), `<b>${f2(r.corrected, 3)}</b>`, f2(r.corrected - R.full, 3)]))}<div class="row gap mt"><button class="primary" id="sm-apply">Reajustar el modelo con estos ${R.kStar} predictores</button><span class="sub">Vuelve al paso 3 y lo ejecuta; se recalcula todo (rendimiento, reporte, código y calculadora).</span></div>`;
      bindFigs(out); $('#sm-apply').addEventListener('click', () => document.dispatchEvent(new CustomEvent('epi:applyvars', { detail: { names: kept.map(orig), from: R.rows.length, tol: R.tol, lab, corrected: cut.corrected, full: R.full } })));
      msg.textContent = '';
    } catch (e) { msg.innerHTML = errBox(e); console.error(e); }
    prog.hidden = true; btn.disabled = false;
  });
}
