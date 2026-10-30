/* EPI-LAB · acceso con usuario y contraseña (barrera de acceso en el navegador).
   Guarda solo una huella PBKDF2-SHA256 de «usuario + contraseña», no la contraseña. Es una barrera para visitantes casuales:
   el código de la herramienta sigue siendo visible para quien sepa abrir las herramientas del navegador.
   Para proteger de verdad, usa acceso en el servidor (Cloudflare Access, Thinkific). Para cambiar la clave: generar_huella.py. */
(function () {
  'use strict';
  var SAL = '4618cfe0ef31e9fc9323f9b5b9c93168', ITER = 200000, HUELLA = 'd3fc21b90b06e3528bdb5e334b8801576a3168dba439318f2ffe7fdae001de9d', KEY = 'epi-lab-acceso', HORAS = 12;
  var d = document, root = d.documentElement;
  function activa() { try { var t = +sessionStorage.getItem(KEY); return t && Date.now() - t < HORAS * 36e5; } catch (e) { return false; } }
  if (activa()) return;
  root.classList.add('epi-bloqueado');
  var st = d.createElement('style');
  st.textContent = 'html.epi-bloqueado body{visibility:hidden!important}' +
    '#epi-gate{position:fixed;inset:0;z-index:2147483647;display:grid;place-items:center;padding:16px;background:#F4F8F9;color:#2B3035;font:400 16px/1.5 "Source Sans 3","Segoe UI",system-ui,Arial,sans-serif;visibility:visible}' +
    '#epi-gate *{box-sizing:border-box}' +
    '#epi-gate form{width:min(100%,380px);background:#fff;border:1px solid #DDE5E8;border-radius:18px;padding:28px;box-shadow:0 18px 50px -24px rgba(31,95,111,.45);display:grid;gap:14px}' +
    '#epi-gate .epi-mk{font:800 1.5rem Montserrat,"Segoe UI",system-ui,Arial,sans-serif;letter-spacing:-.01em}#epi-gate .epi-mk b{color:#2F7F92}' +
    '#epi-gate h1{margin:0;font:700 1.05rem Montserrat,"Segoe UI",system-ui,Arial,sans-serif;color:#8B2E3B}#epi-gate p{margin:0;color:#5E666D;font-size:.95rem}' +
    '#epi-gate label{display:grid;gap:5px;font:600 .82rem Montserrat,"Segoe UI",system-ui,Arial,sans-serif;color:#5E666D}' +
    '#epi-gate input{width:100%;border:1.5px solid #DDE5E8;border-radius:10px;padding:11px 12px;font:inherit;color:#2B3035;background:#fff}' +
    '#epi-gate input:focus{outline:3px solid #AFD5DE;border-color:#2F7F92}' +
    '#epi-gate button{border:0;border-radius:10px;background:#2F7F92;color:#fff;padding:12px 16px;font:700 .95rem Montserrat,"Segoe UI",system-ui,Arial,sans-serif;cursor:pointer}#epi-gate button:disabled{opacity:.6;cursor:wait}' +
    '#epi-gate .epi-err{min-height:1.4em;color:#B3261E;font-size:.92rem}#epi-gate small{color:#5E666D}' +
    '';
  d.head.appendChild(st);
  function bytes(h) { var a = new Uint8Array(h.length / 2); for (var i = 0; i < a.length; i++) a[i] = parseInt(h.substr(i * 2, 2), 16); return a; }
  function hex(b) { return Array.prototype.map.call(new Uint8Array(b), function (x) { return ('0' + x.toString(16)).slice(-2); }).join(''); }
  function huella(u, p) {
    var s = window.crypto && window.crypto.subtle; if (!s) return Promise.reject(new Error('sin-crypto'));
    return s.importKey('raw', new TextEncoder().encode(u.trim().toLowerCase() + '\u0000' + p), 'PBKDF2', false, ['deriveBits'])
      .then(function (k) { return s.deriveBits({ name: 'PBKDF2', hash: 'SHA-256', salt: bytes(SAL), iterations: ITER }, k, 256); }).then(hex);
  }
  function montar() {
    var nombre = (d.title || 'Herramienta').replace(/\s+[·|–—]\s+.*$/, '').trim() || 'Herramienta';
    var g = d.createElement('div'); g.id = 'epi-gate';
    g.innerHTML = '<form autocomplete="on"><div class="epi-mk"><b>EPI</b>-LAB</div><div><h1>Acceso restringido</h1><p>' + nombre.replace(/[<>&]/g, '') + '. Escribe tu usuario y contraseña para continuar.</p></div>' +
      '<label>Usuario<input name="usuario" type="text" autocomplete="username" autocapitalize="none" spellcheck="false" required></label>' +
      '<label>Contraseña<input name="clave" type="password" autocomplete="current-password" required></label>' +
      '<div class="epi-err" role="alert" aria-live="assertive"></div><button type="submit">Entrar</button><small>Tus datos no salen de tu computador.</small></form>';
    root.appendChild(g);
    var f = g.querySelector('form'), err = g.querySelector('.epi-err'), btn = g.querySelector('button'), fallos = 0;
    g.querySelector('input').focus();
    f.addEventListener('submit', function (e) {
      e.preventDefault(); err.textContent = ''; btn.disabled = true;
      var u = f.usuario.value, p = f.clave.value;
      huella(u, p).then(function (h) {
        if (h === HUELLA) { try { sessionStorage.setItem(KEY, String(Date.now())); } catch (x) { } g.remove(); root.classList.remove('epi-bloqueado'); st.remove(); return; }
        fallos++; return new Promise(function (r) { setTimeout(r, Math.min(4000, 700 * fallos)); }).then(function () { err.textContent = 'Usuario o contraseña incorrectos.'; f.clave.value = ''; f.clave.focus(); });
      }).catch(function (x) { err.textContent = x && x.message === 'sin-crypto' ? 'Este navegador no permite verificar la clave aquí. Abre la página con https.' : 'No se pudo verificar. Inténtalo de nuevo.'; })
        .then(function () { btn.disabled = false; });
    });
  }
  if (d.body) montar(); else d.addEventListener('DOMContentLoaded', montar);
})();
