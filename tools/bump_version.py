# Uso: python3 tools/bump_version.py AAAA.MM.DD-N
# Añade ?v=<versión> a todos los imports de js/, a la hoja de estilos y al pie de página (evita cachés antiguas).
import re, sys, pathlib
v = sys.argv[1]; root = pathlib.Path(__file__).resolve().parent.parent
for f in (root / 'js').glob('*.js'):
    s = f.read_text(encoding='utf-8')
    s = re.sub(r"(from '\./[\w-]+\.js)(\?v=[^']*)?'", rf"\1?v={v}'", s)
    f.write_text(s, encoding='utf-8')
h = root / 'index.html'; s = h.read_text(encoding='utf-8')
s = re.sub(r'(css/style\.css|js/app\.js)(\?v=[^"]*)?"', rf'\1?v={v}"', s)
s = re.sub(r'· v[\d.\-]+</footer>', f'· v{v}</footer>', s)
h.write_text(s, encoding='utf-8'); print('versión', v)
