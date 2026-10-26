// Tabla 1 descriptiva, tablas exportables (Word/CSV) y texto de métodos.
import { isMiss, levelsOf } from './data.js?v=2026.10.26-1';
import { shapiro, chisqTest, fisher2x2, welch, anova, mannWhitney, kruskal, mean, sd, median, quantile } from './tests.js?v=2026.10.26-1';

const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export const fp = p => (!isFinite(p) ? '' : p < 0.001 ? '<0.001' : p.toFixed(3));
const f = (v, d = 1) => (isFinite(v) ? v.toFixed(d) : '–');

// ---------------- Tabla 1 ----------------
export function table1(ds, meta, vars, group = null, opts = {}) {
  const en = opts.lang === 'en', G = group ? levelsOf(meta[group]) : [], gcol = group ? ds.cols[group] : null, T = en ? { t: 't-test (Welch)', mw: 'Mann–Whitney U', an: 'ANOVA', kw: 'Kruskal–Wallis', fi: 'Fisher exact', cs: 'Pearson χ²', csc: 'χ² (caution: expected < 5)' } : { t: 't de Welch', mw: 'U de Mann-Whitney', an: 'ANOVA', kw: 'Kruskal-Wallis', fi: 'Exacta de Fisher', cs: 'χ² de Pearson', csc: 'χ² (precaución: esperados < 5)' };
  const idxAll = Array.from({ length: ds.n }, (_, i) => i).filter(i => !group || !isMiss(gcol[i]));
  const byG = G.map(g => idxAll.filter(i => gcol[i] === g));
  const rows = [], head = [en ? 'Variable' : 'Variable', `Total (n = ${idxAll.length})`, ...G.map((g, k) => `${g} (n = ${byG[k].length})`), ...(group ? [en ? 'p-value' : 'Valor p', en ? 'Test' : 'Prueba'] : [])];
  for (const v of vars.filter(v => v !== group)) {
    const m = meta[v], col = ds.cols[v];
    const nmiss = idxAll.filter(i => isMiss(col[i])).length;
    if (m.type === 'num') {
      const all = idxAll.map(i => col[i]).filter(isFinite), sw = shapiro(all.slice(0, 5000)), normal = !opts.nonparam && (sw.p >= 0.05 || (all.length > 200 && Math.abs(skew(all)) < 1));
      const desc = a => (a.length ? (normal ? `${f(mean(a))} ± ${f(sd(a))}` : en ? `${f(median(a))} (${f(quantile(a, 0.25))}–${f(quantile(a, 0.75))})` : `${f(median(a))} [${f(quantile(a, 0.25))}–${f(quantile(a, 0.75))}]`) : '–');
      const gs = byG.map(ix => ix.map(i => col[i]).filter(isFinite));
      let p = NaN, test = '';
      if (group && gs.filter(g => g.length > 1).length >= 2) {
        const ok = gs.filter(g => g.length > 1);
        if (normal) { if (ok.length === 2) { p = welch(ok[0], ok[1]).p; test = T.t; } else { p = anova(ok).p; test = T.an; } }
        else if (ok.length === 2) { p = mannWhitney(ok[0], ok[1]).p; test = T.mw; } else { p = kruskal(ok).p; test = T.kw; }
      }
      rows.push({ cells: [`${v}, ${normal ? (en ? 'mean ± SD' : 'media ± DE') : en ? 'median (IQR)' : 'mediana [RIC]'}`, desc(all), ...gs.map(desc), ...(group ? [fp(p), test] : [])], bold: true, nmiss });
    } else {
      const L = levelsOf(m), cnt = ix => L.map(l => ix.filter(i => col[i] === l).length);
      const tot = cnt(idxAll), gc = byG.map(cnt);
      let p = NaN, test = '';
      if (group) {
        const tab = L.map((_, a) => gc.map(g => g[a])).filter(r => r.some(x => x > 0));
        if (tab.length > 1) {
          const c = chisqTest(tab);
          if (tab.length === 2 && tab[0].length === 2 && c.minE < 5) { p = fisher2x2(tab).p; test = T.fi; } else { p = c.p; test = c.propSmall > 0.2 ? T.csc : T.cs; }
        }
      }
      rows.push({ cells: [`${v}, n (%)`, '', ...G.map(() => ''), ...(group ? [fp(p), test] : [])], bold: true, nmiss });
      const sT = tot.reduce((a, b) => a + b, 0);
      L.forEach((l, a) => rows.push({ cells: [`   ${l}`, `${tot[a]} (${f(100 * tot[a] / sT)})`, ...gc.map(g => { const s = g.reduce((x, y) => x + y, 0); return `${g[a]} (${f(100 * g[a] / (s || 1))})`; }), ...(group ? ['', ''] : [])] }));
    }
  }
  return { head, rows, en, note: en ? `Quantitative variables: mean ± SD when compatible with normality (Shapiro–Wilk p ≥ 0.05)${opts.nonparam ? ' (not applied: all shown as median)' : ''}, otherwise median (interquartile range). ${group ? 'Column percentages. ' : ''}Percentages exclude missing data.` : `Variables cuantitativas: media ± DE si la distribución es compatible con normalidad (Shapiro-Wilk p ≥ 0.05), si no mediana [RIC]. ${group ? 'Porcentajes por columna. ' : ''}Los porcentajes excluyen datos perdidos.` };
}
function skew(a) { const m = mean(a), s = sd(a); return a.reduce((t, v) => t + ((v - m) / s) ** 3, 0) / a.length; }

export const table1HTML = T => `<table class="data t1"><thead><tr>${T.head.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${T.rows.map(r => `<tr>${r.cells.map((c, k) => `<td${r.bold && k === 0 ? ' class="b"' : ''}>${esc(c)}${k === 0 && r.nmiss ? ` <span class="muted">(${T.en ? 'missing' : 'perdidos'}: ${r.nmiss})</span>` : ''}</td>`).join('')}</tr>`).join('')}</tbody></table><p class="notes">${esc(T.note)}</p>`;

// ---------------- tabla de resultados (estilo artículo) ----------------
export function wordDoc(title, tablesHTML, notes = '') {
  return `<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word"><head><meta charset="utf-8"><title>${esc(title)}</title><style>body{font-family:Arial,sans-serif;font-size:10pt}table{border-collapse:collapse;margin:8pt 0}th,td{border-top:1px solid #000;border-bottom:1px solid #000;padding:3pt 6pt;font-size:9pt;text-align:left}th{font-weight:bold}td.b{font-weight:bold}p.n{font-size:8pt}</style></head><body><h3>${esc(title)}</h3>${tablesHTML}${notes ? `<p class="n">${esc(notes)}</p>` : ''}</body></html>`;
}
export const matrixHTML = ({ head, body }) => `<table><thead><tr>${head.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${body.map(r => `<tr>${r.c.map((c, k) => `<td${r.b ? ' class="b"' : ''}>${esc(String(c)).replace(/^ {3}/, '&nbsp;&nbsp;&nbsp;')}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
export const matrixCSV = ({ head, body }) => [head, ...body.map(r => r.c)].map(r => r.map(c => { const s = String(c ?? '').trim(); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; }).join(',')).join('\n');

