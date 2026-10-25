# Bases SIMULADAS de «otro centro» para practicar la validación externa y la recalibración de cada ejemplo de desarrollo.
# Cada base tiene las mismas columnas que su ejemplo de desarrollo, pero otra mezcla de pacientes, otro riesgo basal y efectos algo menores
# (el modelo desarrollado en el ejemplo sobrestima o subestima el riesgo y es demasiado extremo). Ningún dato es real.
# Uso: python3 tools/make_validation_data.py   (requiere numpy, pandas, scipy, statsmodels y lifelines)
import numpy as np, pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from lifelines import WeibullAFTFitter

def copula(df, cols, n, rng):
    """Covariables nuevas con los mismos marginales y correlaciones (cópula gaussiana); la perdida de datos se repone al final."""
    Z = np.zeros((len(df), len(cols))); spec = []
    for j, c in enumerate(cols):
        x = df[c]; ok = x.notna().values
        if x.dtype == object:
            cats = sorted(x[ok].unique(), key=lambda v: (-(x[ok] == v).sum(), v)); p = np.array([(x[ok] == k).mean() for k in cats]); cum = np.concatenate([[0], np.cumsum(p)])
            u = np.full(len(df), 0.5); idx = x[ok].map({k: i for i, k in enumerate(cats)}).values; u[ok] = cum[idx] + p[idx] * rng.random(ok.sum()); spec.append(('cat', cats, cum))
        else:
            r = x[ok].rank(method='average').values; u = np.full(len(df), 0.5); u[ok] = (r - 0.5) / ok.sum(); spec.append(('num', np.sort(x[ok].values), x[ok]))
        Z[:, j] = norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
    S = rng.multivariate_normal(np.zeros(len(cols)), np.corrcoef(Z.T), n); U = norm.cdf(S); out = {}
    for j, c in enumerate(cols):
        kind, a, b = spec[j]
        if kind == 'cat': out[c] = np.array(a, dtype=object)[np.clip(np.searchsorted(b, U[:, j], side='right') - 1, 0, len(a) - 1)]
        else:
            v = np.quantile(a, U[:, j]); dec = 0 if np.all(a == np.round(a)) else max(len(str(s).rstrip('0').split('.')[1]) if '.' in str(s).rstrip('0') else 0 for s in df[c].dropna().head(300)); out[c] = np.round(v, dec)
    return pd.DataFrame(out)
def design(d, cols, ref=None):
    X = pd.get_dummies(d[cols], drop_first=True, dtype=float) if ref is None else pd.get_dummies(d[cols], drop_first=True, dtype=float).reindex(columns=ref.columns, fill_value=0.0)
    return X
def put_missing(d, df, cols, rng):
    for c in cols:
        r = df[c].isna().mean()
        if r > 0: d.loc[rng.random(len(d)) < r, c] = np.nan
    return d
def finish(d, order, name):
    d.insert(0, 'ID', np.arange(1, len(d) + 1)) if 'ID' in order else None; d = d[order]; d.to_csv(f'examples/{name}', index=False); return d

def lesion(n=600, seed=11):
    rng = np.random.default_rng(seed); df = pd.read_csv('examples/lesion_renal_aguda.csv'); cov = ['Edad', 'Sexo', 'Diabetes', 'TFGe basal (mL/min/1.73 m²)', 'Fracción de eyección (%)', 'Tipo de cirugía', 'Tiempo de bypass (min)', 'Vasopresores']
    d = copula(df, cov, n, rng); d['Edad'] = np.clip(d['Edad'] + 4, 25, 92); d['Tiempo de bypass (min)'] = np.round(d['Tiempo de bypass (min)'] * 1.08)
    d['Bypass prolongado (> 120 min)'] = np.where(d['Tiempo de bypass (min)'] > 120, 'Sí', 'No')
    Xd = design(df.assign(**{'Bypass prolongado (> 120 min)': df['Bypass prolongado (> 120 min)']}), cov + ['Bypass prolongado (> 120 min)']).fillna(design(df, cov + ['Bypass prolongado (> 120 min)']).median())
    f = sm.Logit((df['LRA (KDIGO)'] == 'Sí').astype(float), sm.add_constant(Xd)).fit(disp=0)
    Xn = design(d, cov + ['Bypass prolongado (> 120 min)'], Xd); Xn = Xn.fillna(Xd.median()); lp = f.predict(sm.add_constant(Xn, has_constant='add')) ; lp = np.log(lp / (1 - lp))
    lp = 0.80 * lp; lo, hi = -3.0, 3.0
    for _ in range(50): mid = (lo + hi) / 2; lo, hi = (mid, hi) if np.mean(1 / (1 + np.exp(-(lp + mid)))) < 0.21 else (lo, mid)   # LRA ≈ 21 % (en el desarrollo, 14 %)
    d['LRA (KDIGO)'] = np.where(rng.random(n) < 1 / (1 + np.exp(-(lp + lo))), 'Sí', 'No'); d = put_missing(d, df, ['TFGe basal (mL/min/1.73 m²)', 'Fracción de eyección (%)'], rng)
    return finish(d, list(df.columns), 'lesion_renal_aguda_validacion.csv')

def mortalidad(n=500, seed=12):
    rng = np.random.default_rng(seed); df = pd.read_csv('examples/hemodialisis_mortalidad.csv'); cov = ['Acceso vascular', 'Edad', 'Sexo', 'Diabetes', 'Índice de Charlson', 'Albúmina (g/dL)', 'Hemoglobina (g/dL)']
    d = copula(df, cov, n, rng); d['Edad'] = np.clip(d['Edad'] + 3, 18, 92); d['Índice de Charlson'] = np.clip(np.round(d['Índice de Charlson']).astype(int), 0, 12)
    T, E = 'Tiempo de seguimiento (meses)', 'Fallecido'; Xd = design(df, cov); med = Xd.median(); Xd = Xd.fillna(med)
    m = Xd.assign(**{T: df[T].values, E: (df[E] == 'Sí').astype(int).values}); aft = WeibullAFTFitter().fit(m, duration_col=T, event_col=E)
    Xn = design(d, cov, Xd).fillna(med); lam = np.exp(aft.predict_expectation(Xn) * 0 + (Xn.values @ aft.params_['lambda_'][Xd.columns].values + aft.params_['lambda_']['Intercept'])); rho = np.exp(aft.params_['rho_']['Intercept'])
    ll = np.log(lam); ll = 0.80 * (ll - ll.mean()) + ll.mean() + 0.25                       # menos riesgo basal y efectos más débiles que en el desarrollo
    t = np.exp(ll) * (-np.log(rng.random(n))) ** (1 / rho); c = rng.uniform(4, 60, n); d[T] = np.round(np.minimum(t, c), 1).clip(0.2); d[E] = np.where(t <= c, 'Sí', 'No')
    d = put_missing(d, df, ['Albúmina (g/dL)'], rng); return finish(d, list(df.columns), 'hemodialisis_mortalidad_validacion.csv')

def depresion(n=300, seed=13):
    rng = np.random.default_rng(seed); df = pd.read_csv('examples/hemodialisis_depresion.csv'); cov = ['Edad', 'Sexo', 'Diabetes', 'Acceso vascular', 'Tiempo en hemodiálisis (meses)', 'Apoyo familiar', 'Hemoglobina (g/dL)', 'Albúmina (g/dL)', 'Kt/V']
    d = copula(df, cov, n, rng); d['Edad'] = np.clip(d['Edad'] - 3, 18, 90); Xd = design(df, cov); med = Xd.median(); Xd = Xd.fillna(med); y = df['Puntaje PHQ-9']; ok = y.notna()
    f = sm.OLS(y[ok], sm.add_constant(Xd[ok])).fit(); Xn = design(d, cov, Xd).fillna(med); pr = f.predict(sm.add_constant(Xn, has_constant='add')); sd = np.sqrt(f.scale)
    e = rng.normal(0, sd * 1.05, n); lo, hi = -5.0, 8.0
    for _ in range(50): mid = (lo + hi) / 2; lo, hi = (mid, hi) if np.mean(np.clip(0.80 * pr + mid + e, 0, 27) >= 10) < 0.34 else (lo, mid)      # depresión ≈ 34 % (en el desarrollo, 29 %)
    d['Puntaje PHQ-9'] = np.round(np.clip(0.80 * pr + lo + e, 0, 27)); thr = 10
    d['Depresión'] = np.where(d['Puntaje PHQ-9'] >= thr, 'Sí', 'No'); d['Grupo de edad'] = np.where(d['Edad'] < 40, '18 a 39 años', np.where(d['Edad'] < 60, '40 a 59 años', '≥ 60 años'))
    d = put_missing(d, df, ['Apoyo familiar', 'Albúmina (g/dL)', 'Kt/V', 'Puntaje PHQ-9'], rng); return finish(d, list(df.columns), 'hemodialisis_depresion_validacion.csv')

def krt5(n=1500, seed=2031):
    rng = np.random.default_rng(seed)
    etio = rng.choice(['Vascular', 'Diabetic', 'Glomerular', 'Tubulointerstitial', 'Polycystic', 'Unclassified'], n, p=[.32, .26, .15, .15, .05, .07])         # otra mezcla de etiologías
    age = np.clip(rng.normal(73, 12, n) - 17 * (etio == 'Polycystic') - 8 * (etio == 'Glomerular'), 19, 98).round(); sex = np.where(rng.random(n) < .55, 'Male', 'Female')
    diab = np.where((etio == 'Diabetic') | (rng.random(n) < .14), 'Yes', 'No'); htn = np.where(rng.random(n) < np.where(etio == 'Vascular', .95, .72), 'Yes', 'No')
    egfr = np.clip(rng.lognormal(np.log(38), .40, n) + 8 * (etio == 'Polycystic'), 6, 110).round(1); uacr = np.exp(rng.normal(np.log(150), 1.6, n) + 1.1 * (etio == 'Glomerular') + .6 * (etio == 'Diabetic') - .5 * (etio == 'Polycystic')).round(1)
    hb = np.clip(rng.normal(12.2, 1.7, n) - .035 * (60 - egfr) / 5, 6.5, 17).round(1); alb = np.clip(rng.normal(4.0, .45, n) - .25 * (np.log(uacr) > 6.2), 2.0, 5.2).round(2)
    lp0 = - 0.052 * (egfr - 40) + 0.40 * np.log(uacr / 100) - 0.012 * (age - 70) + 0.15 * (sex == 'Male') + 0.25 * (diab == 'Yes') - 0.12 * (hb - 12.5) - 0.5 * (alb - 4.1) + 0.9 * (etio == 'Polycystic') + 0.15 * (etio == 'Glomerular')
    lp0 = 0.85 * lp0; w = rng.weibull(1.25, n); cens = np.minimum(5.0, rng.uniform(1.2, 9.0, n))
    def rate(b0): t = np.exp(-(b0 + lp0) / 1.25) * w; return np.mean(t <= cens)
    lo, hi = -15.0, 5.0
    for _ in range(60): mid = (lo + hi) / 2; lo, hi = (mid, hi) if rate(mid) < 0.10 else (lo, mid)        # TRR a 5 años ≈ 10 % (en el desarrollo, 13 %)
    t = np.exp(-(lo + lp0) / 1.25) * w; time = np.minimum(np.minimum(t, cens).round(2), 5.0); event = (t <= cens) & (time <= 5.0)
    uacr_s = uacr.astype(object); uacr_s[rng.random(n) < .12] = np.nan
    d = pd.DataFrame({'Age (years)': age.astype(int), 'Sex': sex, 'Diabetes': diab, 'Hypertension': htn, 'CKD etiology': etio, 'eGFR (mL/min/1.73 m2)': egfr, 'Albuminuria, UACR (mg/g)': uacr_s, 'Haemoglobin (g/dL)': hb, 'Serum albumin (g/dL)': alb,
                      'Follow-up (years)': time, 'Kidney replacement therapy': np.where(event, 'Yes', 'No'), 'KRT within 5 years': np.where(event, 'Yes', 'No')})
    d.to_csv('examples/erc_krt5_validacion.csv', index=False); return d

if __name__ == '__main__':
    for fn in (lesion, mortalidad, depresion, krt5):
        d = fn(); print(fn.__name__, d.shape, {c: round(float((d[c] == v).mean()), 3) for c, v in [('LRA (KDIGO)', 'Sí'), ('Fallecido', 'Sí'), ('Depresión', 'Sí'), ('KRT within 5 years', 'Yes')] if c in d})
