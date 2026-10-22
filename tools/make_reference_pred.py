# Referencia para tests/validate.mjs: AUC, Brier, calibración (statsmodels/sklearn) y C de Harrell + riesgos (lifelines).
import json, numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
rng = np.random.default_rng(11); n = 700
x1, x2, x3 = rng.normal(size=(3, n)); g = rng.integers(0, 2, n)
eta = -1.2 + 0.9 * x1 - 0.5 * x2 + 0.6 * g
y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
Xb = sm.add_constant(np.column_stack([x1, x2, x3, g])); fit = sm.GLM(y, Xb, family=sm.families.Binomial()).fit(); lp = Xb @ fit.params; p = 1 / (1 + np.exp(-lp))
cal = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
citl = sm.GLM(y, np.ones((n, 1)), family=sm.families.Binomial(), offset=lp).fit()
bin = dict(x1=x1.tolist(), x2=x2.tolist(), x3=x3.tolist(), g=g.tolist(), y=y.tolist(), beta=fit.params.tolist(), auc=roc_auc_score(y, p), brier=brier_score_loss(y, p), slope=cal.params[1], citl=float(citl.params[0]), oe=float(y.sum() / p.sum()))
# supervivencia con empates en el tiempo (tiempos redondeados)
lam = 0.1 * np.exp(0.7 * x1 - 0.5 * g + 0.3 * x2); t = rng.exponential(1 / lam); c = rng.uniform(2, 14, n)
time = np.minimum(t, c).round(1) + 0.1; ev = (t <= c).astype(int)
df = pd.DataFrame(dict(T=time, E=ev, x1=x1, x2=x2, g=g)); cph = CoxPHFitter().fit(df, 'T', 'E')
lpc = cph.predict_log_partial_hazard(df).values
S = cph.predict_survival_function(df.iloc[:5], times=[3.0, 6.0]).values.T  # 5 × 2
surv = dict(time=time.tolist(), event=ev.tolist(), x1=x1.tolist(), x2=x2.tolist(), g=g.tolist(), beta=cph.params_.tolist(), cindex=float(concordance_index(time, -lpc, ev)), surv5=S.tolist(), nev=int(ev.sum()))
json.dump(dict(bin=bin, surv=surv), open('tests/reference_pred.json', 'w'))
print(bin['auc'], surv['cindex'])
