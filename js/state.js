// Estado compartido y utilidades de la interfaz.
export const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export const fp = p => (!isFinite(p) ? '–' : p < 0.001 ? '<0.001' : p.toFixed(3));
export const f2 = (x, d = 2) => (isFinite(x) ? (Math.abs(x) >= 10000 ? x.toFixed(0) : x.toFixed(d)) : '–');
export const pct = (x, d = 1) => (isFinite(x) ? (100 * x).toFixed(d) + '%' : '–');
export const ci = (e, lo, hi, d = 2) => `${f2(e, d)} (${f2(lo, d)} a ${f2(hi, d)})`;
export const opt = (v, label, sel) => `<option value="${esc(v)}"${sel ? ' selected' : ''}>${esc(label ?? v)}</option>`;
export const errBox = e => `<div class="err">${esc(e.message || e)}</div>`;
export const howto = (a, b, c, t = ['Qué se ve', 'Para qué', 'Cómo interpretar']) => `<div class="howto"><div><b>${t[0]}</b>${a}</div><div><b>${t[1]}</b>${b}</div><div><b>${t[2]}</b>${c}</div></div>`;
export const card = (title, body) => `<div class="card"><h2>${title}</h2>${body}</div>`;
export const table = (head, rows, cls = 'data') => `<div class="tablewrap"><table class="${cls}"><thead><tr>${head.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((c, k) => `<td${k && /^[-−\d.<>%(]/.test(String(c)) ? ' class="num"' : ''}>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
export const natural = (a, b) => String(a).localeCompare(String(b), 'es', { numeric: true });
export const state = { raw: [], meta: {}, ds: null, fname: '', sheets: null, spec: null, model: null, final: null, boot: null, shrunk: null, val: null, ss: {}, vmode: 'data', eqTerms: [{ name: '', coef: '' }], rows: null };
export const download = (name, content, type = 'text/plain') => { const blob = content instanceof Blob ? content : new Blob([content], { type }), a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500); };
