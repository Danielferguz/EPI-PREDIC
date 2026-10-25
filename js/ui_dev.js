// Pestaña 4: modelo desarrollado, rendimiento aparente y corregido por bootstrap, calibración y ecuación.
import { $, $$, esc, f2, fp, pct, ci, table, howto, card, state, errBox, download } from './state.js?v=2026.10.25-5';
import { evaluate, shrinkModel, calibGroups, calibSurvival, lowess, riskFromLP, H0at } from './predmodel.js?v=2026.10.25-5';
import { waldTerm, ciOf } from './models.js?v=2026.10.25-5';
import { calibrationPlot, scatterCalib, rocPlot, rocPoints, cvPlot } from './pplots.js?v=2026.10.25-5';
import { figWrap, bindFigs } from './figs.js?v=2026.10.25-5';
import { qnorm } from './dist.js?v=2026.10.25-5';
import { moreCards, bindMore } from './ui_more.js?v=2026.10.25-5';

export function bindFigure() { document.addEventListener('click', () => 0); }
const LAB = {
  binary: { auc: 'Estadístico C (AUC)', slope: 'Pendiente de calibración', citl: 'Calibración en general (CITL)', brier: 'Brier', r2cs: 'R² de Cox-Snell' },
  survival: { cindex: 'Índice C de Harrell', slope: 'Pendiente de calibración (índice pronóstico)', r2cs: 'R² de Cox-Snell' },
  continuous: { r2: 'R²', slope: 'Pendiente de calibración', rmse: 'RMSE' },
};
const IDEAL = { auc: '', slope: '1', citl: '0', brier: '', r2cs: '', cindex: '', r2: '', rmse: '' };
export const modelName = m => ({ binary: 'logístico', survival: 'de Cox', continuous: 'lineal' }[m.type]);
export const expo = m => m.type === 'binary' ? ['OR', 'Razón de odds'] : m.type === 'survival' ? ['HR', 'Razón de riesgos'] : ['β', 'Coeficiente'];

export function coefTable(m) {
  const hasSE = !!m.fit, off = m.type === 'survival' ? 0 : 1, z = qnorm(0.975), [E] = expo(m), rows = [];
  if (m.type !== 'survival') rows.push(['(Intercepto)', f2(m.intercept, 3), hasSE ? f2(m.se[0], 3) : '–', '', hasSE ? fp(m.fit.p[0]) : '']);
  const byVar = {}; m.cols.forEach((c, j) => (byVar[c.v] ||= []).push(j));
  for (const [v, idx] of Object.entries(byVar)) {
    const spl = idx.filter(j => m.cols[j].spline), chunk = hasSE && idx.length > 1 ? waldTerm(m.fit, idx.map(j => j + off)) : null, nonlin = hasSE && spl.length ? waldTerm(m.fit, spl.map(j => j + off)) : null;
    if (idx.length > 1) rows.push([`<b>${esc(v)}</b>`, '', '', spl.length ? `<span class="sub">prueba global p = ${fp(chunk.p)}; no linealidad p = ${fp(nonlin.p)}</span>` : (chunk ? `<span class="sub">prueba global p = ${fp(chunk.p)}</span>` : ''), '']);
    idx.forEach(j => {
      const b = m.beta[j], se = hasSE ? m.fit.se[j + off] : NaN, lab = idx.length > 1 ? '&nbsp;&nbsp;' + esc(m.cols[j].name.replace(v + ': ', '')) : esc(m.cols[j].name), eff = m.type === 'continuous' ? ci(b, b - z * se, b + z * se, 3) : `${f2(Math.exp(b), 2)} (${f2(Math.exp(b - z * se), 2)} a ${f2(Math.exp(b + z * se), 2)})`;
      rows.push([lab, f2(b, 4), hasSE ? f2(se, 4) : '–', hasSE ? (m.cols[j].spline ? '<span class="sub">término de spline</span>' : eff) : (m.type === 'continuous' ? '' : f2(Math.exp(b), 3)), hasSE ? fp(m.fit.p[j + off]) : '']);
    });
  }
  return { head: ['Variable', 'β', 'EE', hasSE ? `${E} (IC 95 %)` : E, 'p'], rows };
}
export function equationText(m, horizon) {
  const t = (b, nm) => `${b < 0 ? '−' : '+'} ${Math.abs(b).toPrecision(4)}·${nm}`, terms = m.cols.map((c, j) => t(m.beta[j], c.name.replace(/\s*\(spline (\d)\)/, ' [spline $1]'))).join('\n  ');
  if (m.type === 'binary') return `logit(p) = ${m.intercept.toPrecision(4)}\n  ${terms}\n\nRiesgo = 1 / (1 + exp(−logit(p)))`;
  if (m.type === 'continuous') return `Ŷ = ${m.intercept.toPrecision(4)}\n  ${terms}\n\nDE residual = ${m.sigma?.toPrecision(4) ?? '–'}`;
  const h = horizon || m.spec.horizon, H = H0at(m.base, h);
  return `PI = ${terms.replace(/^\+ /, '')}\n\nRiesgo(t = ${h}) = 1 − exp(−H₀(${h}) · exp(PI − ${m.lpMean.toPrecision(4)})),   H₀(${h}) = ${H.toPrecision(4)}  [S₀(${h}) = ${Math.exp(-H).toPrecision(4)} en PI = ${m.lpMean.toPrecision(4)}]\nLas variables transformadas con spline requieren los nudos: ${m.info.vars.filter(v => v.knots).map(v => `${v.name}: ${v.knots.map(k => +k.toPrecision(5)).join(', ')}`).join('; ') || '—'}`;
}
const splineNote = m => (m.info.vars.some(v => v.knots) ? `<p class="sub">Nudos de los splines cúbicos restringidos: ${m.info.vars.filter(v => v.knots).map(v => `<b>${esc(v.name)}</b> (${v.knots.map(k => +k.toPrecision(5)).join(', ')})`).join('; ')}.</p>` : '');

export function renderDev() {
  const m0 = state.model, spec = state.spec, out = $('#dev-out'); if (!m0) { out.innerHTML = ''; return; }
  const boot = state.boot, type = m0.type, lab = LAB[type], gamma = boot?.shrinkage, canShrink = !!boot && isFinite(gamma) && !(m0.pen && m0.pen.lambda);
  const m = state.final, ev = evaluate(m, state.dsX, state.metaX, state.dev.rows, { horizon: spec.horizon }), p = m.rows.length, np = m0.cols.length, events = m0.events;
  state.dev.app = ev;
  const kp = `<div class="kpis"><div class="kpi"><b>${m0.n}</b><span>pacientes con datos completos</span></div>${events != null ? `<div class="kpi"><b>${events}</b><span>eventos</span></div>` : ''}<div class="kpi"><b>${np}</b><span>parámetros</span></div>${events != null ? `<div class="kpi"><b>${f2(events / np, 1)}</b><span>eventos por parámetro</span></div>` : ''}${boot ? `<div class="kpi"><b>${f2(gamma, 3)}</b><span>factor de contracción (bootstrap)</span></div>` : ''}${m0.pen?.lambda ? `<div class="kpi"><b>${f2(m0.pen.lambda, 4)}</b><span>λ (${spec.penalty}, ${m0.pen.rule}); ${m0.pen.df} coef. ≠ 0</span></div>` : ''}</div>`;
  const shr = canShrink ? `<label class="inline"><input type="checkbox" id="d-shrink"${state.shrunk ? ' checked' : ''}> Aplicar contracción uniforme (γ = ${f2(gamma, 3)}) al modelo final: los coeficientes se multiplican por γ y se reestima el intercepto (o el riesgo basal)</label>` : '';
  const ct = coefTable(m);
  // rendimiento
  const rows = boot ? boot.rows.map(r => [lab[r.key] || r.key, f2(r.apparent, 3), f2(r.optimism, 3), `<b>${f2(r.corrected, 3)}</b>`, IDEAL[r.key] ?? '']) : Object.keys(lab).map(k => [lab[k], f2(ev.perf[k], 3), '–', '–', IDEAL[k] ?? '']);
  const pf = ev.perf, extra = type === 'binary' ? `<p class="sub">Aparente: C = ${ci(pf.auc, pf.aucLo, pf.aucHi, 3)}; Brier escalado = ${f2(pf.brierScaled, 3)}; R² de Nagelkerke = ${f2(pf.r2nag, 3)}; O/E = ${f2(pf.oe, 3)}.</p>` : type === 'survival' ? `<p class="sub">Aparente: C = ${f2(pf.cindex, 3)} (${pf.events} eventos); riesgo observado/esperado a ${spec.horizon}: ${pct(pf.observed)} / ${pct(pf.expected)} (O/E = ${f2(pf.oe, 2)}).</p>` : `<p class="sub">Aparente: R² = ${f2(pf.r2, 3)}; RMSE = ${f2(pf.rmse, 3)}; MAE = ${f2(pf.mae, 3)}.</p>`;
  // gráficos
  let plots = '';
  if (type === 'binary') {
    const curve = boot?.calibration?.corrected, g = calibGroups(ev.p, ev.y, 10), lw = lowess(ev.p, ev.y, 2 / 3, 40).map(([a, b]) => [a, Math.min(1, Math.max(0, b))]);
    plots += figWrap(calibrationPlot({ groups: g, curve: curve || lw, pred: ev.p, y: ev.y, title: boot?.calibration ? 'Calibration (bootstrap optimism-corrected)' : 'Apparent calibration', note: boot?.calibration ? 'Line: optimism-corrected lowess · dots: apparent risk deciles' : 'Line: lowess · dots: risk deciles' }), 'calibracion_desarrollo');
    plots += figWrap(rocPlot(rocPoints(ev.p, ev.y), `C = ${f2(pf.auc, 3)}`, 'Apparent ROC curve'), 'roc_desarrollo');
  } else if (type === 'survival') {
    const g = calibSurvival(ev.p, ev.time, ev.status, spec.horizon, 5), mx = Math.min(1, Math.max(0.05, ...g.map(r => Math.max(r.pred, r.hi))) * 1.1);
    plots += figWrap(calibrationPlot({ groups: g, curve: null, pred: ev.p, title: `Calibration at ${spec.horizon} (risk quintiles)`, note: 'Observed = Kaplan–Meier by group, 95% CI', max: mx }), 'calibracion_cox_desarrollo');
  } else plots += figWrap(scatterCalib({ pred: ev.p, y: ev.y, curve: lowess(ev.p, ev.y, 2 / 3, 40), title: 'Observed vs predicted' }), 'calibracion_lineal_desarrollo');
  if (m0.pen?.cv?.lambdas) plots += figWrap(cvPlot({ x: m0.pen.cv.lambdas.map(Math.log), y: m0.pen.cv.cv, se: m0.pen.cv.se, xmark: Math.log(m0.pen.cv.lambdas[m0.pen.cv.imin]), xmark2: Math.log(m0.pen.cv.lambdas[m0.pen.cv.i1]) }), 'cv_lambda');
  else if (m0.pen?.cv?.grid) plots += figWrap(cvPlot({ x: m0.pen.cv.grid.map(Math.log), y: m0.pen.cv.score.map(v => -v), xmark: Math.log(m0.pen.lambda), title: 'Choice of λ (Cox ridge)', ylab: '−cross-validated partial log-likelihood' }), 'cv_lambda_cox');
  out.innerHTML = [
    card('Modelo desarrollado', `${howto(`Un modelo ${modelName(m0)} con ${np} parámetros ajustado a ${m0.n} pacientes${events != null ? ` (${events} eventos)` : ''}.`, 'Los coeficientes definen la ecuación de predicción; los splines modelan relaciones no lineales sin categorizar.', 'No interprete los coeficientes como efectos causales (la «falacia de la Tabla 2»): son aportes a la predicción.', ['Qué se ve', 'Para qué', 'Advertencia'])}${kp}${shr ? `<div class="explain">${shr}</div>` : ''}${m0.sep ? '<div class="warn">Posible separación completa: los coeficientes pueden ser inestables; considere penalización.</div>' : ''}`),
    card('Coeficientes', `${table(ct.head, ct.rows)}${splineNote(m)}${m0.pen?.lambda ? '<p class="sub">Con penalización no se muestran errores estándar (los intervalos de Wald no son válidos tras la contracción).</p>' : ''}`),
    card('Rendimiento y validación interna', `${howto('Cada medida en los mismos datos (aparente), el <i>optimismo</i> estimado con bootstrap y el valor <i>corregido</i>.', 'Evaluar el modelo en los datos que lo crearon es optimista: el bootstrap repite todo el proceso de modelado (incluidos nudos y elección de λ) en cada réplica y mide cuánto empeora al aplicarlo a los datos originales.', 'Use la columna «corregida» como estimación del rendimiento esperado en nuevos pacientes de la misma población. Una pendiente corregida &lt; 0.9 sugiere sobreajuste.', ['Qué se ve', 'Para qué', 'Cómo interpretar'])}${table(['Medida', 'Aparente', 'Optimismo', 'Corregida (bootstrap)', 'Ideal'], rows)}${boot ? `<p class="sub">${boot.B} réplicas bootstrap válidas${boot.failed ? ` (${boot.failed} descartadas por no converger)` : ''}; factor de contracción uniforme (media de la pendiente en los datos originales) = <b>${f2(boot.shrinkage, 3)}</b>.</p>` : '<p class="sub">Sin bootstrap: solo se muestra el rendimiento aparente (optimista).</p>'}${extra}${type === 'binary' ? howto('<b>Calibración</b>: riesgo observado frente al predicho (línea = suavizado, puntos = deciles, histograma = distribución de los riesgos). <b>ROC</b>: sensibilidad frente a 1 − especificidad.', 'Comprobar que los riesgos predichos coinciden con los observados y que el modelo ordena bien a los pacientes.', 'La calibración ideal sigue la diagonal; por encima = el modelo subestima, por debajo = sobrestima. En ROC, cuanto más cerca de la esquina superior izquierda, mejor; el C mide el área (0.5 = azar).') : type === 'survival' ? howto('Riesgo observado (Kaplan-Meier) frente al predicho en cinco grupos de riesgo.', 'Comprobar la calibración del riesgo al horizonte elegido.', 'Los puntos deberían seguir la diagonal; el IC amplio indica pocos eventos en el grupo.') : howto('Valor observado frente al predicho (línea = suavizado).', 'Comprobar el ajuste del modelo lineal.', 'Los puntos deberían agruparse en torno a la diagonal sin curvatura sistemática.')}<div class="plots">${plots}</div>`),
    card('Ecuación del modelo', `<p class="sub">Para publicar (TRIPOD exige la ecuación completa) y para validar el modelo en otras bases.</p><pre class="eq">${esc(equationText(m, spec.horizon))}</pre><div class="row gap"><button class="ghost" id="d-copy">Copiar</button><button class="ghost" id="d-json">Descargar modelo (.json)</button></div>`),
    moreCards(),
    `<div class="stepnav"><button class="ghost" data-go="modelo">← Modelo</button><button class="primary" data-go="validacion">Siguiente: validación externa →</button></div>`,
  ].join('');
  bindFigs(out); bindMore();
  $('#d-copy')?.addEventListener('click', () => navigator.clipboard?.writeText(equationText(m, spec.horizon)));
  $('#d-json')?.addEventListener('click', () => download('modelo_prediccion.json', JSON.stringify(exportModel(m), null, 1), 'application/json'));
  $('#d-shrink')?.addEventListener('change', e => {
    if (e.target.checked) { state.shrunk = shrinkModel(state.model, boot.shrinkage, state.dsX, state.metaX); state.final = state.shrunk; } else { state.shrunk = null; state.final = state.model; }
    renderDev();
  });
}
export function exportModel(m) {
  return { app: 'EPI-PREDIC', version: '2026.10.22', type: m.type, spec: { y: m.spec.y, event: m.spec.event, time: m.spec.time, status: m.spec.status, horizon: m.spec.horizon }, intercept: m.intercept ?? null, columns: m.cols.map((c, j) => ({ name: c.name, variable: c.v, level: c.level ?? null, spline: c.spline ?? null, coef: m.beta[j] })), knots: Object.fromEntries(m.info.vars.filter(v => v.knots).map(v => [v.name, v.knots])), levels: Object.fromEntries(m.info.vars.filter(v => v.kind === 'cat').map(v => [v.name, v.levels])), baseline: m.type === 'survival' ? { lpMean: m.lpMean, horizon: m.spec.horizon, H0: H0at(m.base, m.spec.horizon) } : null, sigma: m.sigma ?? null, shrinkage: m.shrunk ?? null, penalty: m.penalty };
}
