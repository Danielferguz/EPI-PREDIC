// Lectura de CSV/XLSX y descarga de archivos.

export function parseCSV(text) {
  text = text.replace(/^﻿/, '');
  const first = text.split(/\r?\n/)[0] || '';
  const delim = (first.match(/;/g) || []).length > (first.match(/,/g) || []).length ? ';' : (first.includes('\t') ? '\t' : ',');
  const rows = []; let row = [], cur = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else q = false; }
      else cur += ch;
    } else if (ch === '"') q = true;
    else if (ch === delim) { row.push(cur); cur = ''; }
    else if (ch === '\n' || ch === '\r') { if (ch === '\r' && text[i + 1] === '\n') i++; row.push(cur); rows.push(row); row = []; cur = ''; }
    else cur += ch;
  }
  if (cur !== '' || row.length) { row.push(cur); rows.push(row); }
  const head = uniqueNames(rows.shift().map(h => h.trim()));
  return rows.filter(r => r.some(x => x.trim() !== '')).map(r => Object.fromEntries(head.map((h, i) => [h, (r[i] ?? '').trim()])));
}
function uniqueNames(h) { const seen = {}; return h.map((x, i) => { let k = x || `V${i + 1}`; if (seen[k]) k = `${k}_${++seen[k]}`; else seen[k] = 1; return k; }); }

export async function readFile(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith('.xlsx') || name.endsWith('.xls')) {
    if (!window.XLSX) throw new Error('No se pudo cargar la librería de Excel (SheetJS). Guarde el archivo como CSV.');
    const wb = window.XLSX.read(await file.arrayBuffer(), { type: 'array', cellDates: true });
    const sheets = {};
    for (const sn of wb.SheetNames) sheets[sn] = window.XLSX.utils.sheet_to_json(wb.Sheets[sn], { defval: '', raw: true });
    return { sheets };
  }
  return { sheets: { [file.name]: parseCSV(await file.text()) } };
}

export function download(name, content, type = 'text/plain') {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}

export const toCSV = (rows, cols) => [cols.join(','), ...rows.map(r => cols.map(c => { const v = r[c] ?? ''; return /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v; }).join(','))].join('\n');
