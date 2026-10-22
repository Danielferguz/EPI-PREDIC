// Exportación de figuras para revistas: SVG vectorial, PNG y TIFF con resolución (DPI) incrustada,
// ancho de columna estándar, fondo blanco, texto negro y fuente sans-serif (Arial/Helvetica).
export const FIG_WIDTHS = { single: 85, onehalf: 120, double: 174 };
const FONT = { arial: 'Arial, Helvetica, sans-serif', helvetica: 'Helvetica, Arial, sans-serif', times: '"Times New Roman", Times, serif' };

export function figSettings() {
  const g = id => document.getElementById(id);
  const w = g('fig-width')?.value || 'double';
  return { mm: w === 'custom' ? +g('fig-mm').value || 174 : FIG_WIDTHS[w], dpi: +(g('fig-dpi')?.value || 600), font: FONT[g('fig-font')?.value || 'arial'] };
}
// tamaño de letra efectivo (pt) de un texto de 12 unidades en una figura de ancho vbW
export const effPt = (vbW, mm) => (12 * mm / vbW) / 0.3528;

function prepare(svg, { mm, font }) {
  const c = svg.cloneNode(true);
  const [, , w, h] = (c.getAttribute('viewBox') || `0 0 ${svg.clientWidth} ${svg.clientHeight}`).split(/\s+/).map(Number);
  c.setAttribute('width', `${mm}mm`); c.setAttribute('height', `${(mm * h / w).toFixed(2)}mm`);
  c.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  c.removeAttribute('class');
  let s = new XMLSerializer().serializeToString(c);
  s = s.replace(/currentColor/g, '#000000').replace(/var\(--muted,\s*#666\)/g, '#4D4D4D')
    .replace(/font:\s*([\d.]+)px\s+system-ui,\s*sans-serif/g, (m0, px) => `font-size:${px}px;font-family:${font}`)
    .replace(/font:(\d+)px system-ui,sans-serif/g, (m0, px) => `font-size:${px}px;font-family:${font}`);
  // fondo blanco
  s = s.replace(/(<svg[^>]*>)/, `$1<rect x="0" y="0" width="${w}" height="${h}" fill="#FFFFFF"/>`);
  return { s, w, h };
}

const crcTable = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
const crc32 = b => { let c = 0xFFFFFFFF; for (let i = 0; i < b.length; i++) c = crcTable[(c ^ b[i]) & 0xFF] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
// inserta el bloque pHYs (píxeles por metro) tras IHDR para que el PNG declare sus DPI
function pngWithDpi(buf, dpi) {
  const src = new Uint8Array(buf), ppm = Math.round(dpi / 0.0254);
  const chunk = new Uint8Array(21), dv = new DataView(chunk.buffer);
  dv.setUint32(0, 9); chunk.set([0x70, 0x48, 0x59, 0x73], 4); dv.setUint32(8, ppm); dv.setUint32(12, ppm); chunk[16] = 1;
  dv.setUint32(17, crc32(chunk.subarray(4, 17)));
  const at = 8 + 25; // firma + IHDR
  const out = new Uint8Array(src.length + 21); out.set(src.subarray(0, at)); out.set(chunk, at); out.set(src.subarray(at), at + 21);
  return out;
}
// TIFF RGB sin compresión con resolución
function tiff(img, dpi) {
  const { width: W, height: H, data } = img, n = W * H * 3, ifdN = 11, hdr = 8, ifdSize = 2 + ifdN * 12 + 4;
  const bpsOff = hdr + ifdSize, resOff = bpsOff + 6, dataOff = resOff + 16;
  const buf = new ArrayBuffer(dataOff + n), dv = new DataView(buf), u8 = new Uint8Array(buf);
  u8.set([0x49, 0x49]); dv.setUint16(2, 42, true); dv.setUint32(4, hdr, true);
  let p = hdr; dv.setUint16(p, ifdN, true); p += 2;
  const ent = (tag, type, count, val) => { dv.setUint16(p, tag, true); dv.setUint16(p + 2, type, true); dv.setUint32(p + 4, count, true); if (type === 3 && count === 1) dv.setUint16(p + 8, val, true); else dv.setUint32(p + 8, val, true); p += 12; };
  ent(256, 4, 1, W); ent(257, 4, 1, H); ent(258, 3, 3, bpsOff); ent(259, 3, 1, 1); ent(262, 3, 1, 2); ent(273, 4, 1, dataOff);
  ent(277, 3, 1, 3); ent(278, 4, 1, H); ent(279, 4, 1, n); ent(282, 5, 1, resOff); ent(283, 5, 1, resOff + 8);
  dv.setUint32(p, 0, true);
  [8, 8, 8].forEach((v, i) => dv.setUint16(bpsOff + 2 * i, v, true));
  dv.setUint32(resOff, dpi, true); dv.setUint32(resOff + 4, 1, true); dv.setUint32(resOff + 8, dpi, true); dv.setUint32(resOff + 12, 1, true);
  // ResolutionUnit (296) = pulgadas es el valor por defecto en TIFF
  for (let i = 0, j = dataOff; i < W * H; i++) { u8[j++] = data[4 * i]; u8[j++] = data[4 * i + 1]; u8[j++] = data[4 * i + 2]; }
  return u8;
}
function save(name, blob) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 800);
}
export async function exportFigure(svg, name, fmt) {
  const set = figSettings(), { s, w, h } = prepare(svg, set);
  const base = name.replace(/[^\w\-]+/g, '_').replace(/_+/g, '_').slice(0, 80) || 'figure';
  if (fmt === 'svg') return save(`${base}.svg`, new Blob([s], { type: 'image/svg+xml' }));
  const pxW = Math.round(set.mm / 25.4 * set.dpi), pxH = Math.round(pxW * h / w);
  const url = URL.createObjectURL(new Blob([s], { type: 'image/svg+xml' }));
  const img = new Image(); img.decoding = 'sync';
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url; });
  const cv = document.createElement('canvas'); cv.width = pxW; cv.height = pxH;
  const ctx = cv.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, pxW, pxH); ctx.drawImage(img, 0, 0, pxW, pxH);
  URL.revokeObjectURL(url);
  if (fmt === 'png') {
    const blob = await new Promise(r => cv.toBlob(r, 'image/png'));
    return save(`${base}_${set.dpi}dpi.png`, new Blob([pngWithDpi(await blob.arrayBuffer(), set.dpi)], { type: 'image/png' }));
  }
  return save(`${base}_${set.dpi}dpi.tif`, new Blob([tiff(ctx.getImageData(0, 0, pxW, pxH), set.dpi)], { type: 'image/tiff' }));
}
// envuelve un SVG con la barra de descarga
export const figWrap = (svg, name) => `<figure class="fig" data-name="${String(name).replace(/"/g, '')}">${svg}<figcaption class="figbar"><span>Descargar:</span><button type="button" data-fig="svg">SVG</button><button type="button" data-fig="png">PNG</button><button type="button" data-fig="tiff">TIFF</button></figcaption></figure>`;
export function bindFigs(root) {
  root.querySelectorAll('figure.fig:not([data-bound])').forEach(f => (f.dataset.bound = '1') && f.querySelectorAll('[data-fig]').forEach(b => b.addEventListener('click', async () => {
    const svg = f.querySelector('svg'); const t = b.textContent; b.textContent = '…';
    try { await exportFigure(svg, f.dataset.name, b.dataset.fig); } catch (e) { alert('No se pudo exportar: ' + e.message); }
    b.textContent = t;
  })));
}
