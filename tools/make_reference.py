# Genera tests/reference_samplesize.json con el paquete de Python `pmsampsize` (puerto de Riley et al.).
# Uso: pip install pmsampsize numpy scipy statsmodels pandas tabulate && python3 tools/make_reference.py
import io, json, contextlib, itertools
from pmsampsize import pmsampsize as m

def run(**k):
    with contextlib.redirect_stdout(io.StringIO()):
        return m.pmsampsize(noprint=True, **k)
out = {'binary': [], 'survival': [], 'continuous': [], 'cstat': []}
for r2, p, prev in itertools.product([0.05, 0.1, 0.2, 0.3], [5, 15, 30], [0.05, 0.2, 0.5]):
    try:
        r = run(type='b', csrsquared=r2, parameters=p, prevalence=prev)
        out['binary'].append(dict(r2cs=r2, p=p, prev=prev, crit=[row[1] for row in r['results_table'] if row != m.SEPARATING_LINE][:3], n=r['sample_size'], epp=r['EPP'], max_r2=r['max_r2a']))
    except Exception as e: pass
for r2, p, rate, tp, fu in [(0.051, 30, 0.065, 2, 2.07), (0.1, 10, 0.05, 5, 3), (0.2, 20, 0.2, 1, 1.5), (0.03, 12, 0.02, 10, 6), (0.15, 8, 0.1, 3, 4)]:
    r = run(type='s', csrsquared=r2, parameters=p, rate=rate, timepoint=tp, meanfup=fu)
    out['survival'].append(dict(r2cs=r2, p=p, rate=rate, timepoint=tp, meanfup=fu, crit=[row[1] for row in r['results_table'] if row != m.SEPARATING_LINE][:3], n=r['sample_size'], epp=r['EPP'], max_r2=r['max_r2a']))
for r2, p in itertools.product([0.1, 0.2, 0.5, 0.7], [5, 25, 60]):
    r = run(type='c', rsquared=r2, parameters=p, intercept=50, sd=5)  # intercepto grande: el criterio 4 no es el que decide
    out['continuous'].append(dict(r2=r2, p=p, crit=[row[1] for row in r['results_table'] if row != m.SEPARATING_LINE][:4], n=r['sample_size']))
for c, prev in [(0.7, 0.5), (0.8, 0.5), (0.89, 0.5), (0.75, 0.3), (0.85, 0.1), (0.65, 0.2)]:
    # el puerto de Python asigna la proporción `prev` a los NO eventos; para una prevalencia de eventos `q` se pasa 1 − q
    out['cstat'].append(dict(c=c, prev_events=prev, r2cs=float(m.cstat2rsq(c, 1 - prev, 123456, True)['R2_coxsnell'])))
json.dump(out, open('tests/reference_samplesize.json', 'w'), indent=1)
print({k: len(v) for k, v in out.items()})
