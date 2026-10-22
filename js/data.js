// Tipado de variables, conjuntos de datos y matrices de diseño (codificación de tratamiento, interacciones).

export const NA_DEFAULT = ['', 'na', 'n/a', 'nan', 'null', '.', '-', 'missing', 'perdido', 'sin dato', 's/d', 'nd'];

const toNum = v => {
  if (typeof v === 'number') return v;
  const s = String(v).trim().replace(/\s/g, '');
  if (!/^[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?$/.test(s)) return NaN;
  return parseFloat(s.replace(',', '.'));
};
const natural = (a, b) => { const x = toNum(a), y = toNum(b); return isFinite(x) && isFinite(y) ? x - y : String(a).localeCompare(String(b), 'es', { numeric: true }); };

// detecta el tipo de cada columna: num (cuantitativa), cat (categórica), id (identificador)
export function inferMeta(rows, naCodes = NA_DEFAULT) {
  const names = Object.keys(rows[0] || {}), meta = {}, na = new Set(naCodes.map(s => String(s).toLowerCase().trim()));
  for (const name of names) {
    const vals = rows.map(r => r[name]).filter(v => v != null && !na.has(String(v).toLowerCase().trim()));
    const nums = vals.map(toNum), allNum = vals.length > 0 && nums.every(isFinite);
    const distinct = [...new Set(vals.map(v => String(v).trim()))];
    let type = 'cat';
    // numérica: más de 6 valores, o 3–6 valores que no parecen códigos (enteros en un rango corto)
    if (allNum && (distinct.length > 6 || (distinct.length > 2 && (nums.some(v => !Number.isInteger(v)) || Math.max(...nums) - Math.min(...nums) > 6)))) type = 'num';
    if (!allNum && distinct.length > Math.max(25, 0.5 * vals.length)) type = 'id';
    if (/^(id|codigo|código|cod|record|registro|participante|paciente|subject|patient)(_?id)?$/i.test(name)) type = 'id';
    const levels = distinct.sort(natural);
    meta[name] = { name, type, levels, ref: levels[0], event: levels.length === 2 ? levels[1] : levels[levels.length - 1], nDistinct: distinct.length, allNum };
  }
  return meta;
}

// convierte las filas en columnas tipadas: num → Float64Array con NaN, cat → string o null
export function typedData(rows, meta, naCodes = NA_DEFAULT) {
  const na = new Set(naCodes.map(s => String(s).toLowerCase().trim())), cols = {};
  for (const [name, m] of Object.entries(meta)) {
    const raw = rows.map(r => r[name]);
    if (m.type === 'num') cols[name] = Float64Array.from(raw, v => (v == null || na.has(String(v).toLowerCase().trim()) ? NaN : toNum(v)));
    else cols[name] = raw.map(v => (v == null || na.has(String(v).toLowerCase().trim()) ? null : String(v).trim()));
  }
  return { n: rows.length, cols };
}
export const isMiss = v => v == null || (typeof v === 'number' && !isFinite(v));
export const missCount = col => { let k = 0; for (let i = 0; i < col.length; i++) if (isMiss(col[i])) k++; return k; };

// orden de niveles con la referencia primero
export const levelsOf = m => [m.ref, ...m.levels.filter(l => l !== m.ref)];

// filas completas para un conjunto de variables
export function completeRows(ds, vars, base = null) {
  const out = [];
  for (let i = 0; i < ds.n; i++) { if (base && !base[i]) continue; if (vars.every(v => !isMiss(ds.cols[v][i]))) out.push(i); }
  return out;
}

// términos: [{vars:['edad']}, {vars:['sexo']}, {vars:['sexo','edad']} (interacción)]
// devuelve X (con intercepto si intercept), columnas con etiquetas y índices por término
export function design(ds, meta, terms, rows, { intercept = true } = {}) {
  const blocks = terms.map(t => t.vars.map(v => varBlock(ds, meta[v], rows)));
  const cols = [], termIdx = [];
  if (intercept) cols.push({ term: '(Intercept)', label: '(Intercepto)', vars: [], levels: [] });
  const colVals = intercept ? [rows.map(() => 1)] : [];
  terms.forEach((t, k) => {
    // producto cartesiano de las columnas de cada variable del término
    let acc = [{ vals: rows.map(() => 1), names: [], levels: [] }];
    blocks[k].forEach((b, j) => {
      const next = [];
      for (const a of acc) for (const c of b) next.push({ vals: a.vals.map((v, i) => v * c.vals[i]), names: [...a.names, c.name], levels: [...a.levels, c.level] });
      acc = next;
    });
    const idx = [];
    for (const c of acc) { idx.push(cols.length); cols.push({ term: termName(t), label: c.names.join(' × '), vars: t.vars, levels: c.levels }); colVals.push(c.vals); }
    termIdx.push({ name: termName(t), vars: t.vars, idx });
  });
  const X = rows.map((_, i) => Float64Array.from(colVals, c => c[i]));
  return { X, cols, terms: termIdx };
}
export const termName = t => t.vars.join(':');
function varBlock(ds, m, rows) {
  const col = ds.cols[m.name];
  if (m.type === 'num') return [{ name: m.name, level: null, vals: rows.map(i => col[i]) }];
  return levelsOf(m).slice(1).map(l => ({ name: `${m.name}: ${l}`, level: l, vals: rows.map(i => (col[i] === l ? 1 : 0)) }));
}

// variable de respuesta: num → valor; binaria → 1 si es el nivel de evento
export function response(ds, m, rows) {
  const col = ds.cols[m.name];
  if (m.type === 'num') return Float64Array.from(rows, i => col[i]);
  return Float64Array.from(rows, i => (String(col[i]) === String(m.event) ? 1 : 0));
}

// recuentos por nivel (para la columna n de las tablas)
export function levelCounts(ds, m, rows) {
  const col = ds.cols[m.name], o = {};
  for (const i of rows) o[col[i]] = (o[col[i]] || 0) + 1;
  return o;
}
