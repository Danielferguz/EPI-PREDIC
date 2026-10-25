// Bases de práctica para la validación externa: cada ejemplo de desarrollo tiene una base simulada de «otro centro» con las mismas columnas
// (generadas con tools/make_validation_data.py; ningún dato es real). dev = nombre del ejemplo de desarrollo sin extensión.
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export const PRACTICE = [
  { dev: 'erc_prediccion_desarrollo', file: 'erc_prediccion_validacion.csv', title: 'ERC · falla renal a 2 años (logístico y Cox)', n: 900, what: 'Pacientes 4 años mayores y con TFGe 5 menor; mayor riesgo basal y efectos 15 % menores: el modelo subestima el riesgo y es demasiado extremo.' },
  { dev: 'erc_krt5_manuscrito', file: 'erc_krt5_validacion.csv', title: 'Manuscrito · TRR a 5 años (logístico y Cox)', n: 1500, what: 'Más ERC diabética, pacientes mayores y con menor TFGe; TRR a 5 años 10 % (13 % en el desarrollo) y efectos 15 % menores: el modelo sobrestima el riesgo.' },
  { dev: 'lesion_renal_aguda', file: 'lesion_renal_aguda_validacion.csv', title: 'Lesión renal aguda poscirugía cardiaca', n: 600, what: 'Pacientes 4 años mayores y bypass más largo; LRA 22 % (14 %) y efectos 20 % menores.' },
  { dev: 'hemodialisis_mortalidad', file: 'hemodialisis_mortalidad_validacion.csv', title: 'Hemodiálisis · mortalidad (Cox)', n: 500, what: 'Pacientes 3 años mayores; mortalidad algo menor y efectos 20 % más débiles que en el desarrollo.' },
  { dev: 'hemodialisis_depresion', file: 'hemodialisis_depresion_validacion.csv', title: 'Hemodiálisis · puntaje PHQ-9 (lineal)', n: 300, what: 'Pacientes 3 años más jóvenes y PHQ-9 más alto (depresión 37 % frente a 29 %); pendiente 0.8 respecto al desarrollo.' },
];
export const practiceFor = fname => PRACTICE.find(p => p.dev === fname);
export function practiceHTML(fname) {
  const cur = practiceFor(fname);
  return `<details class="mt" open><summary><b>Bases de práctica</b> para esta etapa (simuladas, de «otro centro»)</summary>
    <p class="sub">${cur ? `Para el ejemplo cargado use <b>${esc(cur.file)}</b>: ` + esc(cur.what) : 'Cada base solo sirve para el ejemplo de desarrollo con el que se emparejó (mismas columnas). Cargue primero ese ejemplo en el paso 2 y ajuste el modelo.'}</p>
    <div class="tablewrap"><table class="data"><thead><tr><th>Ejemplo de desarrollo</th><th>Base de práctica</th><th>Qué cambia respecto al desarrollo</th><th></th></tr></thead><tbody>${PRACTICE.map(p => `<tr${p === cur ? ' style="background:var(--soft)"' : ''}><td>${esc(p.title)}</td><td>${esc(p.file)}<br><span class="sub">${p.n} pacientes</span></td><td>${esc(p.what)}</td><td><button class="ghost sm" data-practice="${esc(p.file)}">Usar</button> <a href="examples/${esc(p.file)}" download>CSV</a></td></tr>`).join('')}</tbody></table></div></details>`;
}
// aviso cuando la base cargada no corresponde al modelo ajustado
export const practiceHint = fname => { const c = practiceFor(fname); return c ? ` Esta base no corresponde al modelo ajustado con «${fname}»: use la base de práctica «${c.file}» (o una base propia con las mismas columnas).` : ''; };
