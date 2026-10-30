// Calculadora de riesgo descargable (HTML autónomo, sin conexión) a partir del modelo exportado (exportModel).
const E = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// Se copia tal cual dentro del HTML. M: modelo exportado; vals: {variable original: valor}; devuelve riesgo (o valor predicho)
export function calcRisk(M, vals) {
  const rcs = (x, t) => { const k = t.length, d = (t[k - 1] - t[0]) ** 2, pos = v => (v > 0 ? v ** 3 : 0), o = [x]; for (let j = 0; j < k - 2; j++) o.push((pos(x - t[j]) - pos(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2]) + pos(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])) / d); return o; };
  let lp = M.intercept || 0;
  for (const c of M.columns) {
    const v = c.variable, ln = /^ln\((.*)\)$/.exec(v), raw = ln ? ln[1] : v; let x;
    if (M.levels[v]) x = String(vals[raw]) === c.level ? 1 : 0;
    else { let a = +vals[raw]; if (ln) a = Math.log(a); if (!isFinite(a)) return NaN; x = M.knots[v] ? rcs(a, M.knots[v])[c.spline || 0] : a; }
    lp += c.coef * x;
  }
  if (M.type === 'binary') return 1 / (1 + Math.exp(-lp));
  if (M.type === 'survival') return 1 - Math.exp(-M.baseline.H0 * Math.exp(lp - M.baseline.lpMean));
  return lp;
}
// info: {title, outcome, vars:[{name, kind:'num'|'cat', levels?, min?, max?, median?}], n, events, horizon, date, perf, notes}
export function calcPageHTML(M, info) {
  const payload = JSON.stringify({ M, vars: info.vars }).replace(/</g, '\\u003c'), bin = M.type === 'binary', sv = M.type === 'survival';
  const fields = info.vars.map((v, i) => (v.kind === 'cat' ? `<label>${E(v.name)}<select data-i="${i}">${v.levels.map(l => `<option>${E(l)}</option>`).join('')}</select></label>` : `<label>${E(v.name)}<input type="number" step="any" data-i="${i}" value="${v.median ?? ''}" placeholder="${v.min != null ? `observado ${E(v.min)}–${E(v.max)}` : ''}"></label>`)).join('');
  return `<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${E(info.title)}</title>
<style>:root{--bg:#fff;--fg:#1f2933;--mu:#6b7785;--line:#d9dee4;--ac:#2E7F90;--card:#f6f8fa}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}main{max-width:720px;margin:0 auto;padding:20px 16px 40px}h1{font-size:1.3rem;margin:.2em 0}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
label{display:flex;flex-direction:column;font-size:.85rem;color:var(--mu);gap:3px}input,select{font:inherit;padding:7px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}button{font:inherit;padding:8px 16px;border:0;border-radius:6px;background:var(--ac);color:#fff;cursor:pointer}.big{font-size:2.2rem;font-weight:700;color:var(--ac)}.mu{color:var(--mu);font-size:.85rem}.warn{border-left:4px solid #B3261E;background:rgba(179,38,30,.08);padding:8px 12px;border-radius:6px;font-size:.85rem}</style></head><body><main>
<h1>${E(info.title)}</h1><p class="mu">${bin ? 'Riesgo predicho' : sv ? `Riesgo predicho a ${E(info.horizon)}` : 'Valor predicho'} de «${E(info.outcome)}». Modelo desarrollado el ${E(info.date)} en ${info.n} pacientes${info.events != null ? ` (${info.events} eventos)` : ''}. Funciona sin conexión.</p>
<div class="warn"><b>Herramienta de investigación.</b> Modelo sin validación externa: no use esta calculadora para decidir la atención de un paciente. Introduzca todos los valores; la estimación solo es válida para pacientes similares a los del desarrollo.${info.perf ? ` Rendimiento interno corregido: ${E(info.perf)}.` : ''}</div>
<div class="card"><div class="grid">${fields}</div><p><button id="go">Calcular</button></p></div>
<div class="card" id="out" hidden><div class="mu" id="lab"></div><div class="big" id="big"></div></div><p class="mu">${E(info.notes || '')}</p>
<script type="application/json" id="model">${payload}</script>
<script>
const calcRisk = ${calcRisk.toString()};
const D = JSON.parse(document.getElementById('model').textContent), $ = s => document.querySelector(s);
$('#go').onclick = () => { const vals = {}; document.querySelectorAll('[data-i]').forEach(el => { vals[D.vars[+el.dataset.i].name] = el.value; }); const r = calcRisk(D.M, vals);
  $('#out').hidden = false; $('#lab').textContent = D.M.type === 'continuous' ? 'Valor predicho' : 'Riesgo predicho'; $('#big').textContent = isNaN(r) ? 'Complete todos los campos' : (D.M.type === 'continuous' ? r.toFixed(2) : (100 * r).toFixed(1) + ' %'); };
</script></main></body></html>`;
}
