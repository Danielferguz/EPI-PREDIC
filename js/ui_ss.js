// Pestaña 1: tamaño de muestra para desarrollo y validación de modelos de predicción.
import { devBinary, devSurvival, devContinuous, valBinary, valContinuous, seFromWidth, maxR2binary, maxR2survival, cstatToR2cs } from './samplesize.js?v=2026.10.25-8';
import { $, $$, esc, f2, table, state, errBox } from './state.js?v=2026.10.25-8';
import { nPlot } from './pplots.js?v=2026.10.25-8';
import { figWrap, bindFigs } from './figs.js?v=2026.10.25-8';

const v = id => parseFloat($(id).value);
const ceil = Math.ceil;

function devInputs() {
  const type = $('#sd-type').value, src = $('#sd-r2src').value, r2v = v('#sd-r2'), o = { type, p: ceil(v('#sd-p')), shrink: v('#sd-S'), delta: v('#sd-delta'), margin: v('#sd-margin') };
  if (type === 'binary') Object.assign(o, { prev: v('#sd-prev') });
  if (type === 'survival') Object.assign(o, { rate: v('#sd-rate'), timepoint: v('#sd-tp'), meanfup: v('#sd-fu') });
  if (type === 'continuous') Object.assign(o, { r2: r2v, intercept: v('#sd-int'), sd: v('#sd-sd') });
  else if (src === 'cs') o.r2cs = r2v; else if (src === 'nag') o.nag = r2v; else o.cstat = r2v;
  return o;
}
export function runDev(o) {
  if (o.type === 'binary') return devBinary(o);
  if (o.type === 'survival') return devSurvival(o);
  return devContinuous(o);
}
function layoutDev() {
  const type = $('#sd-type').value, src = $('#sd-r2src').value;
  $$('.sv').forEach(e => (e.hidden = type !== 'survival')); $$('.ct').forEach(e => (e.hidden = type !== 'continuous')); $$('.nct').forEach(e => (e.hidden = type === 'continuous'));
  $('#sd-prev-w').hidden = type !== 'binary';
  const o = $('#sd-r2src'); [...o.options].forEach(op => (op.hidden = op.value === 'c' && type !== 'binary')); if (type !== 'binary' && o.value === 'c') o.value = 'cs';
  $('#sd-margin').parentElement.hidden = type === 'continuous';
  const lab = type === 'continuous' ? 'R² ajustado esperado' : { cs: 'R² de Cox-Snell esperado', nag: 'R² de Nagelkerke esperado', c: 'Estadístico C esperado' }[$('#sd-r2src').value];
  $('#sd-r2-w').firstChild.textContent = lab + ' '; $('#sd-r2src').parentElement.hidden = type === 'continuous';
  $('#sd-hint').innerHTML = { binary: 'Use el rendimiento de un modelo previo del mismo campo (R² ajustado o C optimismo-corregido); sea conservador. Si solo conoce el C, se convierte en R² de Cox-Snell suponiendo un predictor lineal normal con varianza común.', survival: 'Tasa = eventos por persona-tiempo (p. ej., 0.065 por año); el seguimiento medio y el tiempo de predicción deben estar en las mismas unidades. Para R² de Cox-Snell use el de un modelo previo (rara vez se reporta: estime desde el C o desde la proporción de variación explicada).', continuous: 'Use el R² ajustado de un modelo previo. El intercepto es la media esperada del desenlace y la DE su desviación estándar (para la precisión de la media y de la varianza residual).' }[type];
}
function renderDev() {
  layoutDev(); const out = $('#sd-out');
  try {
    const o = devInputs(), r = runDev(o); state.ss.dev = { input: o, result: r };
    const cs = r.type === 'continuous' ? r.r2 : r.r2cs;
    const head = r.type === 'continuous' ? ['Criterio', 'n', 'Contracción', 'Sujetos por parámetro'] : ['Criterio', 'n', 'Contracción', 'Eventos', 'Eventos por parámetro (EPP)'];
    const rows = r.crit.map((c, i) => [`${i + 1}. ${esc(c.label)}`, `<b${c.n === r.n ? ' class="ok"' : ''}>${c.n.toLocaleString('es')}</b>`, f2(c.shrinkage, 3), ...(r.type === 'continuous' ? [f2(c.n / o.p, 1)] : [f2(c.events, 0), f2(c.epp, 1)])]);
    rows.push([`<b>Tamaño mínimo</b>`, `<b>${r.n.toLocaleString('es')}</b>`, '', ...(r.type === 'continuous' ? [f2(r.n / o.p, 1)] : [`<b>${f2(r.events, 0)}</b>`, `<b>${f2(r.epp, 1)}</b>`])]);
    const info = r.type === 'continuous' ? `R² ajustado = ${f2(r.r2, 3)}` : `R² de Cox-Snell = ${f2(r.r2cs, 3)} (máximo posible ${f2(r.maxR2, 3)}; R² de Nagelkerke = ${f2(r.nagR2, 3)})${o.cstat ? ` · obtenido del C = ${o.cstat}` : ''}`;
    let sens = '';
    if (r.type !== 'continuous') {
      const base = r.r2cs, vals = [0.5, 0.75, 1, 1.25, 1.5].map(k => base * k).filter(x => x < r.maxR2 * 0.98 && x < o.shrink), rowsS = vals.map(x => { const q = runDev({ ...o, r2cs: x, nag: undefined, cstat: undefined }); return [f2(x, 3), q.n.toLocaleString('es'), f2(q.events, 0), f2(q.epp, 1)]; });
      const mk = pp => ({ label: `p = ${pp}`, pts: Array.from({ length: 24 }, (_, k) => { const x = Math.max(0.01, base * 0.4) + (Math.min(r.maxR2 * 0.9, base * 2.2) - Math.max(0.01, base * 0.4)) * k / 23; try { return [x, runDev({ ...o, p: pp, r2cs: x, nag: undefined, cstat: undefined }).n]; } catch { return null; } }).filter(Boolean) });
      sens = `<h3>Sensibilidad al rendimiento esperado</h3>${table(['R² de Cox-Snell', 'n mínimo', 'Eventos', 'EPP'], rowsS)}<div class="plots single">${figWrap(nPlot([mk(Math.max(1, Math.round(o.p / 2))), mk(o.p), mk(o.p * 2)], { mark: [r.r2cs, r.n] }), 'tamano_muestra_vs_R2')}</div>`;
    }
    const lang = es => r.type === 'binary' ? (es
      ? `El tamaño de muestra se calculó con el método de Riley et al. para modelos con desenlace binario (paquete pmsampsize). Con ${o.p} parámetros candidatos, una prevalencia del ${f2(o.prev * 100, 1)} % y un R² de Cox-Snell esperado de ${f2(r.r2cs, 3)}${o.cstat ? ` (equivalente a un estadístico C de ${o.cstat})` : ''}, se requieren al menos ${r.n.toLocaleString('es')} participantes (${f2(r.events, 0)} eventos; ${f2(r.epp, 1)} eventos por parámetro candidato) para lograr un factor de contracción global ≥ ${o.shrink}, una diferencia ≤ ${o.delta} entre el R² de Nagelkerke aparente y ajustado y un margen de error ≤ ${o.margin} en el riesgo global.`
      : `Sample size was calculated following Riley et al. for binary outcomes (pmsampsize). With ${o.p} candidate predictor parameters, an outcome prevalence of ${f2(o.prev * 100, 1)}% and an anticipated Cox-Snell R² of ${f2(r.r2cs, 3)}${o.cstat ? ` (equivalent to a C statistic of ${o.cstat})` : ''}, at least ${r.n.toLocaleString('en')} participants (${f2(r.events, 0)} events; ${f2(r.epp, 1)} events per candidate parameter) are required to target a global shrinkage factor ≥ ${o.shrink}, an absolute difference ≤ ${o.delta} between apparent and adjusted Nagelkerke R², and a margin of error ≤ ${o.margin} in the overall risk.`)
      : r.type === 'survival' ? (es
      ? `El tamaño de muestra se calculó con el método de Riley et al. para desenlaces de supervivencia. Con ${o.p} parámetros candidatos, una tasa global de eventos de ${o.rate} por unidad de tiempo, un seguimiento medio de ${o.meanfup} y un R² de Cox-Snell esperado de ${f2(r.r2cs, 3)}, se requieren al menos ${r.n.toLocaleString('es')} participantes (${f2(r.events, 0)} eventos; ${f2(r.epp, 1)} eventos por parámetro) para un factor de contracción global ≥ ${o.shrink}, una diferencia ≤ ${o.delta} entre R² aparente y ajustado y un margen de error ≤ ${o.margin} en el riesgo acumulado a ${o.timepoint}.`
      : `Sample size followed Riley et al. for time-to-event outcomes. With ${o.p} candidate parameters, an overall event rate of ${o.rate} per time unit, mean follow-up of ${o.meanfup} and an anticipated Cox-Snell R² of ${f2(r.r2cs, 3)}, at least ${r.n.toLocaleString('en')} participants (${f2(r.events, 0)} events; ${f2(r.epp, 1)} events per parameter) are needed for a global shrinkage ≥ ${o.shrink}, a difference ≤ ${o.delta} between apparent and adjusted R², and a margin of error ≤ ${o.margin} in the cumulative risk at ${o.timepoint}.`)
      : (es
      ? `El tamaño de muestra se calculó con el método de Riley et al. para desenlaces continuos. Con ${o.p} parámetros candidatos y un R² ajustado esperado de ${f2(r.r2, 3)}, se requieren al menos ${r.n.toLocaleString('es')} participantes (${f2(r.n / o.p, 1)} por parámetro) para un factor de contracción global ≥ ${o.shrink}, una diferencia ≤ ${o.delta} entre R² aparente y ajustado y una estimación precisa de la varianza residual y del intercepto.`
      : `Sample size followed Riley et al. for continuous outcomes. With ${o.p} candidate parameters and an anticipated adjusted R² of ${f2(r.r2, 3)}, at least ${r.n.toLocaleString('en')} participants (${f2(r.n / o.p, 1)} per parameter) are required for a global shrinkage ≥ ${o.shrink}, a difference ≤ ${o.delta} between apparent and adjusted R², and precise estimates of the residual variance and intercept.`);
    out.innerHTML = `<div class="kpis"><div class="kpi"><b>${r.n.toLocaleString('es')}</b><span>participantes (mínimo)</span></div>${r.type === 'continuous' ? '' : `<div class="kpi"><b>${f2(r.events, 0)}</b><span>eventos necesarios</span></div><div class="kpi"><b>${f2(r.epp, 1)}</b><span>eventos por parámetro</span></div>`}<div class="kpi"><b>${o.p}</b><span>parámetros candidatos</span></div></div><p class="sub">${info}</p>${table(head, rows)}<div class="explain"><p><b>Texto para la sección de métodos</b> <button class="ghost sm" data-cp="es">Copiar ES</button> <button class="ghost sm" data-cp="en">Copy EN</button></p><p id="sd-txt-es">${lang(true)}</p><p id="sd-txt-en" class="muted">${lang(false)}</p></div>${sens}<p class="notes">Referencias: Riley RD et al. <i>Stat Med</i> 2019;38:1276 y 1296; Riley RD et al. <i>BMJ</i> 2020;368:m441. Los criterios reproducen el paquete <code>pmsampsize</code> (validado en <code>tests/validate_samplesize.mjs</code>).</p>`;
    bindFigs(out);
    $$('[data-cp]', out).forEach(b => b.addEventListener('click', () => navigator.clipboard?.writeText($(`#sd-txt-${b.dataset.cp}`).textContent)));
  } catch (e) { out.innerHTML = errBox(e); state.ss.dev = null; }
}

function layoutVal() {
  const t = $('#sv-type').value; $$('.vb').forEach(e => (e.hidden = t !== 'binary')); $$('.vc').forEach(e => (e.hidden = t !== 'continuous'));
  if (t === 'binary') { const n = $('#sv-lpmode').value === 'n'; $$('.lpn').forEach(e => (e.hidden = !n)); $('#sv-c').parentElement.hidden = n; $('#sv-prev').parentElement.hidden = n; }
}
function renderVal() {
  layoutVal(); const out = $('#sv-out'), t = $('#sv-type').value;
  try {
    let r, rows;
    if (t === 'binary') {
      const n = $('#sv-lpmode').value === 'n', pts = $('#sv-pt').value.split(/[,;\s]+/).map(parseFloat).filter(x => x > 0 && x < 1);
      r = valBinary({ prev: v('#sv-prev'), cstat: v('#sv-c'), lpMean: n ? v('#sv-lpm') : NaN, lpSd: n ? v('#sv-lps') : NaN, seOE: seFromWidth(v('#sv-wOE')), seSlope: seFromWidth(v('#sv-wS')), seCstat: seFromWidth(v('#sv-wC')), thresholds: pts });
      rows = r.crit.map(c => [esc(c.label), `<b${c.n === r.n ? ' class="ok"' : ''}>${c.n.toLocaleString('es')}</b>`, f2(c.n * r.phi, 0), esc(c.note || '')]);
      rows.push(['<b>Tamaño mínimo</b>', `<b>${r.n.toLocaleString('es')}</b>`, `<b>${f2(r.events, 0)}</b>`, '']);
      const nbd = r.crit.find(c => c.detail)?.detail;
      out.innerHTML = `<div class="kpis"><div class="kpi"><b>${r.n.toLocaleString('es')}</b><span>participantes (mínimo)</span></div><div class="kpi"><b>${f2(r.events, 0)}</b><span>eventos esperados</span></div><div class="kpi"><b>${f2(r.phi * 100, 1)}%</b><span>prevalencia (implícita)</span></div><div class="kpi"><b>${f2(r.C, 3)}</b><span>estadístico C implícito</span></div></div>${table(['Criterio', 'n', 'Eventos', 'Objetivo'], rows)}${nbd ? `<h3>Beneficio neto estandarizado</h3>${table(['Umbral', 'Sensibilidad', 'Especificidad', 'sNB esperado', 'SE objetivo', 'n'], nbd.map(x => [x.pt, f2(x.se, 3), f2(x.sp, 3), f2(x.snb, 3), f2(x.target, 4), x.n.toLocaleString('es')]))}` : ''}<p class="notes">Se supone un modelo bien calibrado (intercepto 0, pendiente 1). Un valor conservador para la pendiente esperada es 1: si espera sobreajuste (&lt; 1) la precisión cambia poco. Fuente: Riley RD et al. <i>BMJ</i> 2024;384:e074821.</p>`;
    } else {
      r = valContinuous({ r2: v('#sv-r2'), varY: v('#sv-var'), seR2: seFromWidth(v('#sv-wR2')), seCITL: seFromWidth(v('#sv-wCITL')), seSlope: seFromWidth(v('#sv-wSc')) });
      out.innerHTML = `<div class="kpis"><div class="kpi"><b>${r.n.toLocaleString('es')}</b><span>participantes (mínimo)</span></div></div>${table(['Criterio', 'n', 'Objetivo'], [...r.crit.map(c => [esc(c.label), `<b${c.n === r.n ? ' class="ok"' : ''}>${c.n.toLocaleString('es')}</b>`, esc(c.note)]), ['<b>Tamaño mínimo</b>', `<b>${r.n.toLocaleString('es')}</b>`, '']])}<p class="notes">Repita el cálculo con R² ± 0.1 y tome el mayor. Fuente: Riley RD et al. <i>BMJ</i> 2024;384:e074821 (figura 2).</p>`;
    }
    state.ss.val = r;
  } catch (e) { out.innerHTML = errBox(e); }
}

export function initSampleSize() {
  $$('[data-ss]').forEach(b => b.addEventListener('click', () => { $$('[data-ss]').forEach(x => x.classList.toggle('active', x === b)); $('#ss-dev').hidden = b.dataset.ss !== 'dev'; $('#ss-val').hidden = b.dataset.ss !== 'val'; (b.dataset.ss === 'dev' ? renderDev : renderVal)(); }));
  $$('#ss-dev input, #ss-dev select').forEach(e => e.addEventListener('input', renderDev)); $$('#ss-val input, #ss-val select').forEach(e => e.addEventListener('input', renderVal));
  $('#sd-type').addEventListener('change', () => { const t = $('#sd-type').value; $('#sd-r2').value = t === 'continuous' ? 0.4 : t === 'survival' ? 0.051 : 0.2; if (t === 'survival') $('#sd-p').value = 30; renderDev(); });
  renderDev();
}
export const refreshSS = () => { renderDev(); };
