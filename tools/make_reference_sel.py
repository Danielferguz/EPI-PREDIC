# Referencia (statsmodels) de la eliminación hacia atrás por AIC, BIC y QIC → tests/reference_sel.json
import json, numpy as np, pandas as pd, statsmodels.api as sm
d = pd.read_csv('examples/erc_prediccion_desarrollo.csv')
V = ['Edad (años)', 'Sexo', 'Diabetes', 'TFGe (mL/min/1.73 m²)', 'UACR (mg/g)', 'Hemoglobina (g/dL)', 'Bicarbonato (mmol/L)', 'Fosfato (mg/dL)']
d = d.dropna(subset=V + ['Falla renal a 2 años']); y = (d['Falla renal a 2 años'] == 'Sí').astype(float).to_numpy()
def X_of(vs):
    cols = []
    for v in vs: cols += [pd.get_dummies(d[v], drop_first=True).astype(float)] if d[v].dtype == object else [d[[v]].astype(float)]
    return sm.add_constant(pd.concat(cols, axis=1).to_numpy(float))
def crit(vs, c):
    X = X_of(vs); r = sm.Logit(y, X).fit(disp=0)
    if c == 'aic': return r.aic
    if c == 'bic': return r.bic
    mu = r.predict(X); A = (X.T * (mu * (1 - mu))) @ X; B = (X.T * (y - mu) ** 2) @ X
    return -2 * r.llf + 2 * np.trace(np.linalg.solve(A, B))
out = {'vars': V}
for c in ('aic', 'bic', 'qic'):
    cur = list(V); c0 = crit(cur, c); path = []
    while len(cur) > 1:
        best = min(((crit([w for w in cur if w != v], c), v) for v in cur))
        if best[0] >= c0 - 1e-9: break
        cur.remove(best[1]); c0 = best[0]; path.append(best[1])
    out[c] = {'kept': cur, 'removed': path, 'final': c0}
json.dump(out, open('tests/reference_sel.json', 'w'), indent=1, ensure_ascii=False); print({k: v for k, v in out.items() if k != 'vars'})
