// Código Python (statsmodels, scikit-learn, lifelines, SciPy) que replica el análisis, sin código de figuras.
const PY = String.raw`
import json, sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
T = CFG["type"]; KPOS = {3: [.10, .50, .90], 4: [.05, .35, .65, .95], 5: [.05, .275, .5, .725, .95]}
if T == "survival":
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
expit = lambda z: 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ datos y diseño
def load(path):
    df = pd.read_csv(path, na_values=CFG["na_values"])
    for v in CFG["predictors"]:
        if v["type"] == "num":
            df[v["name"]] = pd.to_numeric(df[v["name"]], errors="coerce")
            if v["form"] == "log": df[v["name"]] = np.log(df[v["name"]].where(df[v["name"]] > 0))
    need = [v["name"] for v in CFG["predictors"]] + ([CFG["time"], CFG["status"]] if T == "survival" else [CFG["y"]])
    return df.dropna(subset=need).reset_index(drop=True)


def outcome(df):
    if T == "binary": return (df[CFG["y"]].astype(str) == str(CFG["event"])).astype(float).to_numpy()
    if T == "continuous": return pd.to_numeric(df[CFG["y"]]).to_numpy(float)
    return None


def rcs(x, t):
    k = len(t); d = (t[-1] - t[0]) ** 2; pos = lambda v: np.where(v > 0, v ** 3, 0.0); cols = [x]
    for j in range(k - 2):
        cols.append((pos(x - t[j]) - pos(x - t[k - 2]) * (t[-1] - t[j]) / (t[-1] - t[k - 2]) + pos(x - t[-1]) * (t[k - 2] - t[j]) / (t[-1] - t[k - 2])) / d)
    return cols


def make_info(df, preds):
    """Nudos de los splines y niveles de las categóricas, estimados en df (se repite en cada réplica bootstrap)."""
    info = []
    for v in preds:
        w = dict(v)
        if v["type"] == "num":
            k = int(v["form"][3:]) if v["form"].startswith("rcs") else 0
            x = df[v["name"]].to_numpy(float); w["knots"] = np.quantile(x[np.isfinite(x)], KPOS[k]) if k and len(np.unique(x)) > k + 1 else None
        info.append(w)
    return info


def design(df, info):
    cols, names = [], []
    for v in info:
        if v["type"] == "cat":
            for lv in v["levels"][1:]: cols.append((df[v["name"]].astype(str) == str(lv)).astype(float).to_numpy()); names.append(f"{v['name']}: {lv}")
        elif v.get("knots") is not None:
            for j, c in enumerate(rcs(df[v["name"]].to_numpy(float), v["knots"])): cols.append(c); names.append(v["name"] if j == 0 else f"{v['name']} (spline {j})")
        else: cols.append(df[v["name"]].to_numpy(float)); names.append(v["name"])
    return np.column_stack(cols), names


# ------------------------------------------------------------------ ajuste
class Model:
    def __init__(self, info, names, beta, intercept=0.0, cov=None, fitobj=None, lp_mean=0.0):
        self.info, self.names, self.beta, self.intercept, self.cov, self.fit, self.lp_mean = info, names, np.asarray(beta, float), intercept, cov, fitobj, lp_mean
    def lp(self, df):
        X, _ = design(df, self.info); return self.intercept + X @ self.beta
    def risk(self, df, horizon=None):
        z = self.lp(df)
        if T == "binary": return expit(z)
        if T == "continuous": return z
        return 1 - self.fit.predict_survival_function(pd.DataFrame(design(df, self.info)[0], columns=self.names), times=[horizon]).to_numpy().ravel()


def crit_value(df, preds, crit):
    """AIC, BIC o QIC (varianza robusta, correlación independiente) de un modelo sin penalizar."""
    info = make_info(df, preds); X, _ = design(df, info); y = outcome(df)
    if T == "survival":
        d = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])]); d["_t"] = df[CFG["time"]].to_numpy(float); d["_e"] = (df[CFG["status"]].astype(str) == str(CFG["sevent"])).astype(float).to_numpy()
        ll = CoxPHFitter().fit(d, "_t", "_e").log_likelihood_; return -2 * ll + (np.log(d["_e"].sum()) if crit == "bic" else 2) * X.shape[1]
    Xc = sm.add_constant(X); r = (sm.Logit(y, Xc) if T == "binary" else sm.OLS(y, Xc)).fit(disp=0)
    if crit == "aic": return r.aic
    if crit == "bic": return r.bic
    mu = r.predict(Xc); res = y - mu
    if T == "binary": A = (Xc.T * (mu * (1 - mu))) @ Xc; B = (Xc.T * res ** 2) @ Xc
    else: s2 = np.sum(res ** 2) / len(y); A = Xc.T @ Xc / s2; B = (Xc.T * res ** 2) @ Xc / s2 ** 2
    return -2 * r.llf + 2 * np.trace(np.linalg.solve(A, B))


def select_backward(df, preds, crit):
    """Eliminación hacia atrás de predictores completos: se retira el que más reduce el criterio hasta que ninguno lo mejora."""
    cur = list(preds); c0 = crit_value(df, cur, crit)
    while len(cur) > 1:
        best = None
        for v in cur:
            try: c = crit_value(df, [w for w in cur if w is not v], crit)
            except Exception: continue
            if best is None or c < best[0]: best = (c, v)
        if best is None or best[0] >= c0 - 1e-9: break
        cur.remove(best[1]); c0 = best[0]
    return cur


def fit_model(df, preds, penalty=None, seed=1, lam=None):
    if penalty is None and CFG.get("selection") not in (None, "none"): preds = select_backward(df, preds, CFG["selection"])  # se repite en cada réplica bootstrap
    info = make_info(df, preds); X, names = design(df, info); y = outcome(df); pen = penalty or CFG["penalty"]
    if T == "survival":
        d = pd.DataFrame(X, columns=names); d["_t"] = df[CFG["time"]].to_numpy(float); d["_e"] = (df[CFG["status"]].astype(str) == str(CFG["sevent"])).astype(float).to_numpy()
        if pen == "ridge":
            sc = StandardScaler().fit(X); Z = pd.DataFrame(sc.transform(X), columns=names); Z["_t"], Z["_e"] = d["_t"], d["_e"]
            best, bs = None, -np.inf; rng = np.random.default_rng(seed); fold = rng.integers(0, 10, len(d))
            for l in ([lam] if lam else np.logspace(-3, 1, 12)):
                s = 0
                for k in range(10):
                    try: c = CoxPHFitter(penalizer=l, l1_ratio=0).fit(Z[fold != k], "_t", "_e"); s += c.score(Z[fold == k], scoring_method="log_likelihood") * (fold == k).sum()
                    except Exception: s = -np.inf; break
                if s > bs: best, bs = l, s
            c = CoxPHFitter(penalizer=best, l1_ratio=0).fit(Z, "_t", "_e"); b = c.params_.to_numpy() / sc.scale_
            return Model(info, names, b, 0.0, None, CoxPHFitter(penalizer=best, l1_ratio=0).fit(d, "_t", "_e") if False else c, 0.0), best
        c = CoxPHFitter().fit(d, "_t", "_e"); return Model(info, names, c.params_.to_numpy(), 0.0, c.variance_matrix_.to_numpy(), c), None
    if pen in (None, "none"):
        Xc = sm.add_constant(X); r = (sm.Logit(y, Xc) if T == "binary" else sm.OLS(y, Xc)).fit(disp=0)
        return Model(info, names, r.params[1:], r.params[0], r.cov_params()[1:, 1:], r), None
    sc = StandardScaler().fit(X); Z = sc.transform(X)
    if T == "binary": r = LogisticRegressionCV(Cs=30, cv=10, penalty="l1" if pen == "lasso" else "l2", solver="saga" if pen == "lasso" else "lbfgs", scoring="neg_log_loss", max_iter=5000, random_state=seed).fit(Z, y); b, a = r.coef_[0], r.intercept_[0]
    else: r = (LassoCV(cv=10, random_state=seed) if pen == "lasso" else RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(Z, y); b, a = r.coef_, r.intercept_
    bo = b / sc.scale_; return Model(info, names, bo, a - float(np.sum(bo * sc.mean_))), None


# ------------------------------------------------------------------ rendimiento
def perf(m, df, horizon=None):
    z = m.lp(df)
    if T == "binary":
        y = outcome(df); p = expit(z); n1 = y.sum(); n0 = len(y) - n1; auc = roc_auc_score(y, p)
        sl = sm.Logit(y, sm.add_constant(z)).fit(disp=0).params[1]; citl = sm.GLM(y, np.ones(len(y)), family=sm.families.Binomial(), offset=z).fit().params[0]
        return {"auc": auc, "slope": sl, "citl": citl, "brier": float(np.mean((p - y) ** 2)), "oe": float(y.sum() / p.sum())}
    if T == "continuous":
        y = outcome(df); sse = np.sum((y - z) ** 2); return {"r2": 1 - sse / np.sum((y - y.mean()) ** 2), "slope": sm.OLS(y, sm.add_constant(z)).fit().params[1], "rmse": float(np.sqrt(sse / len(y)))}
    t = df[CFG["time"]].to_numpy(float); e = (df[CFG["status"]].astype(str) == str(CFG["sevent"])).astype(float).to_numpy()
    d = pd.DataFrame({"lp": z, "t": t, "e": e}); sl = CoxPHFitter().fit(d, "t", "e").params_["lp"]
    return {"cindex": concordance_index(t, -z, e), "slope": sl}


KEY = {"binary": "auc", "survival": "cindex", "continuous": "r2"}[T]


def bootstrap_optimism(df, preds, B, seed=1, fixed_lambda=None):
    """Optimismo de Harrell: en cada réplica se repite todo el proceso (nudos, elección de λ y ajuste)."""
    rng = np.random.default_rng(seed); m0, _ = fit_model(df, preds, seed=seed); app = perf(m0, df, CFG.get("horizon")); opt = {k: [] for k in app}
    for b in range(B):
        s = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
        try:
            mb, _ = fit_model(s, preds, seed=seed + b); a = perf(mb, s, CFG.get("horizon")); t = perf(mb, df, CFG.get("horizon"))
            for k in app: opt[k].append(a[k] - t[k])
        except Exception: continue
    res = pd.DataFrame({"medida": list(app), "aparente": list(app.values()), "optimismo": [np.mean(opt[k]) for k in app]}); res["corregida"] = res["aparente"] - res["optimismo"]
    return m0, res


# ------------------------------------------------------------------ Tabla 1 y regresión cruda/ajustada
def table1(df0, raw_preds, group):
    rows = []; g = df0[group]; lv = [x for x in pd.unique(g.dropna())]
    for v in raw_preds:
        x = df0[v["name"]]
        if v["type"] == "num":
            xs = [pd.to_numeric(x[g == l], errors="coerce").dropna() for l in lv]; q = lambda a: f"{a.median():.1f} ({a.quantile(.25):.1f}–{a.quantile(.75):.1f})"
            p = stats.mannwhitneyu(xs[0], xs[1], alternative="two-sided").pvalue if len(lv) == 2 else stats.kruskal(*xs).pvalue
            rows.append([f"{v['name']}, median (IQR)", q(pd.to_numeric(x, errors="coerce").dropna())] + [q(a) for a in xs] + [p])
        else:
            tab = pd.crosstab(x, g); p = stats.fisher_exact(tab.values)[1] if tab.shape == (2, 2) and (stats.contingency.expected_freq(tab.values) < 5).any() else stats.chi2_contingency(tab.values, correction=False)[1]
            rows.append([f"{v['name']}, n (%)", ""] + [""] * len(lv) + [p])
            for l in v["levels"]: rows.append([f"   {l}", f"{(x == l).sum()} ({100 * (x == l).mean():.1f})"] + [f"{((x == l) & (g == k)).sum()} ({100 * ((x == l) & (g == k)).sum() / (g == k).sum():.1f})" for k in lv] + [np.nan])
    return pd.DataFrame(rows, columns=["variable", "total"] + [str(k) for k in lv] + ["p"])


def crude_adjusted(df, preds):
    info = make_info(df, [dict(v, form="lin") for v in preds]); X, names = design(df, info); y = outcome(df); out = []
    def fit(Xm):
        if T == "survival":
            d = pd.DataFrame(Xm); d.columns = [str(i) for i in range(Xm.shape[1])]; d["_t"] = df[CFG["time"]].to_numpy(float); d["_e"] = (df[CFG["status"]].astype(str) == str(CFG["sevent"])).astype(float).to_numpy()
            c = CoxPHFitter().fit(d, "_t", "_e"); return np.exp(c.params_.to_numpy()), np.exp(c.confidence_intervals_.to_numpy()), c.summary["p"].to_numpy()
        r = (sm.Logit(y, sm.add_constant(Xm)) if T == "binary" else sm.OLS(y, sm.add_constant(Xm))).fit(disp=0); ci = r.conf_int(); f = np.exp if T == "binary" else (lambda a: a)
        return f(r.params[1:]), f(ci[1:]), r.pvalues[1:]
    ea, ca, pa = fit(X)
    for j, nm in enumerate(names):
        e, c, p = fit(X[:, [j]]); out.append([nm, e[0], c[0][0], c[0][1], p[0], ea[j], ca[j][0], ca[j][1], pa[j]])
    return pd.DataFrame(out, columns=["variable", "crude", "crude_lo", "crude_hi", "crude_p", "adjusted", "adj_lo", "adj_hi", "adj_p"])


# ------------------------------------------------------------------ simplificación (importancia de Wald parcial y rendimiento corregido por bootstrap)
def rank_vars(df, preds):
    m, _ = fit_model(df, preds, penalty="none"); out = []
    for v in preds:
        idx = [j for j, n in enumerate(m.names) if n == v["name"] or n.startswith(v["name"] + ": ") or n.startswith(v["name"] + " (spline")]
        b = m.beta[idx]; V = m.cov[np.ix_(idx, idx)]; out.append((v["name"], float(b @ np.linalg.solve(V, b)) - len(idx)))
    return [n for n, _ in sorted(out, key=lambda t: -t[1])]


def simplify_path(df, preds, B, tol, seed=1):
    byname = {v["name"]: v for v in preds}; order = rank_vars(df, preds); p = len(order); rng = np.random.default_rng(seed); app, opt = [], [[] for _ in range(p)]
    for k in range(1, p + 1): m, _ = fit_model(df, [byname[n] for n in order[:k]], penalty="none"); app.append(perf(m, df, CFG.get("horizon"))[KEY])
    for b in range(B):
        s = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
        try:
            ob = rank_vars(s, preds)
            for k in range(1, p + 1): m, _ = fit_model(s, [byname[n] for n in ob[:k]], penalty="none"); opt[k - 1].append(perf(m, s, CFG.get("horizon"))[KEY] - perf(m, df, CFG.get("horizon"))[KEY])
        except Exception: continue
    tab = pd.DataFrame({"k": range(1, p + 1), "anadido": order, "aparente": app, "optimismo": [np.mean(o) for o in opt]}); tab["corregido"] = tab["aparente"] - tab["optimismo"]
    kstar = int(tab[tab["corregido"] >= tab["corregido"].iloc[-1] - tol]["k"].min()); return tab, order[:kstar]


# ------------------------------------------------------------------ calibración por grupos y beneficio neto (tablas)
def calibration_table(p, y, g=10):
    d = pd.DataFrame({"p": p, "y": y}); d["grupo"] = pd.qcut(d["p"].rank(method="first"), g, labels=False) + 1
    return d.groupby("grupo").agg(n=("y", "size"), predicho=("p", "mean"), observado=("y", "mean")).reset_index()


def decision_curve(p, y, th=np.arange(0.05, 0.81, 0.05)):
    n = len(y); prev = y.mean(); rows = []
    for t in th: tp = ((p >= t) & (y == 1)).sum(); fp = ((p >= t) & (y == 0)).sum(); w = t / (1 - t); rows.append([t, tp / n - fp / n * w, prev - (1 - prev) * w])
    return pd.DataFrame(rows, columns=["umbral", "beneficio_neto_modelo", "beneficio_neto_tratar_a_todos"])


def main(data_path, ext_path=None):
    raw = pd.read_csv(data_path, na_values=CFG["na_values"]); df = load(data_path); preds = CFG["predictors"]
    print(f"{len(df)} pacientes con datos completos (de {len(raw)})")
    if CFG["simplify"]["run"]:
        tab, keep = simplify_path(df, preds, CFG["simplify"]["B"], CFG["simplify"]["tol"]); tab.to_csv("simplificacion.csv", index=False); print(tab.round(3).to_string(index=False)); preds = [v for v in preds if v["name"] in keep]
    elif CFG["simplify"]["selected"]: preds = [v for v in preds if v["name"] in CFG["simplify"]["selected"]]
    grp = CFG["y"] if T == "binary" else CFG.get("status")
    if grp: table1(raw.dropna(subset=[grp]), [dict(v) for v in CFG["predictors_raw"]], grp).to_csv("tabla1.csv", index=False)
    crude_adjusted(df, preds).to_csv("regresion_cruda_ajustada.csv", index=False)
    m, res = bootstrap_optimism(df, preds, CFG["B"], CFG["seed"]); print(res.round(3).to_string(index=False)); res.to_csv("rendimiento_bootstrap.csv", index=False)
    print("Variables del modelo final:", [v["name"] for v in m.info])
    pd.DataFrame({"variable": m.names, "coeficiente": m.beta}).assign(intercepto=m.intercept).to_csv("coeficientes.csv", index=False)
    if T == "binary": p = m.risk(df); calibration_table(p, outcome(df)).to_csv("calibracion_por_grupos.csv", index=False); decision_curve(p, outcome(df)).to_csv("curva_decision.csv", index=False)
    if ext_path:  # validación externa: aplicar el modelo sin reajustarlo
        ext = load(ext_path); print("Validación externa:", {k: round(float(v), 3) for k, v in perf(m, ext, CFG.get("horizon")).items()})


if __name__ == "__main__":
    # uso: python replicar.py desarrollo.csv [validacion.csv]
    main(sys.argv[1] if len(sys.argv) > 1 else CFG["data_file"], sys.argv[2] if len(sys.argv) > 2 else None)
`;

// cfg: {data_file, na_values, type, y, event, time, status, sevent, horizon, penalty, B, seed, predictors:[{name,type,form,levels}], predictors_raw, simplify:{run,B,tol,selected}, meta:{n,events,date}}
export function pythonScript(cfg) {
  const head = `#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Código de replicación · modelo de predicción clínica (${cfg.type === 'binary' ? 'regresión logística' : cfg.type === 'survival' ? 'regresión de Cox' : 'regresión lineal'}) · generado el ${cfg.meta.date}
Desenlace: ${cfg.type === 'survival' ? `${cfg.status} (evento: ${cfg.sevent}; tiempo: ${cfg.time}; horizonte ${cfg.horizon})` : `${cfg.y}${cfg.type === 'binary' ? ` (evento: ${cfg.event})` : ''}`} · ${cfg.meta.n} pacientes · ${cfg.predictors.length} predictores.

Requisitos: Python >= 3.9, numpy, pandas, scipy, statsmodels, scikit-learn${cfg.type === 'survival' ? ', lifelines' : ''}.   Uso:  python replicar.py desarrollo.csv [validacion.csv]

El script reproduce, sin código de figuras: Tabla 1 por desenlace, regresión cruda y ajustada por predictor, el modelo final (splines cúbicos
restringidos de Harrell con los mismos nudos), la selección hacia atrás por AIC/BIC/QIC si se usó (repetida en cada réplica), la corrección del optimismo por bootstrap (repitiendo todo el proceso), la simplificación del modelo
por importancia de Wald parcial, tablas de calibración por grupos y de beneficio neto, y la validación externa. La aplicación calcula con una implementación
propia, verificada frente a statsmodels, scikit-learn y lifelines; el bootstrap usa otro generador aleatorio, y la elección de λ en modelos penalizados
(rejilla y regla del mínimo) puede diferir, por lo que se esperan diferencias pequeñas, no idénticas.
"""
`;
  return `${head}\nimport json\nCFG = json.loads(r'''${JSON.stringify(cfg, null, 1).replace(/'''/g, "'")}''')\n${PY}`;
}
