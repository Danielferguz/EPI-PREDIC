# Genera datos SIMULADOS de nefrología para EPI-PREDIC (desarrollo y validación externa de un modelo de falla renal).
# Uso: python3 tools/make_examples.py   (requiere numpy y pandas)
import numpy as np, pandas as pd

B0 = {}
def simulate(n, seed, shift=False):
    r = np.random.default_rng(seed)
    edad = np.clip(r.normal(63 + (4 if shift else 0), 13, n), 20, 92).round(0)
    sexo = r.random(n) < 0.55
    dm = r.random(n) < (0.45 if shift else 0.40) + 0.002 * (edad - 63)
    hta = r.random(n) < 0.82
    tfg = np.clip(r.normal(40 - (5 if shift else 0), 14, n) - 0.08 * (edad - 63), 8, 85).round(1)
    uacr = np.exp(r.normal(5.6 + 0.4 * dm + 0.1 * (tfg < 30), 1.5, n)).round(0).clip(3, 8000)
    hb = np.clip(r.normal(12.4, 1.6, n) + 0.045 * (tfg - 40) - 0.3 * dm, 6.5, 17).round(1)
    bic = np.clip(r.normal(24.2, 3.0, n) + 0.03 * (tfg - 40), 12, 34).round(1)
    fos = np.clip(r.normal(3.7, 0.65, n) - 0.012 * (tfg - 40), 2.0, 7.5).round(1)
    alb = np.clip(r.normal(4.0, 0.45, n) - 0.15 * dm, 2.0, 5.2).round(1)
    pas = np.clip(r.normal(136, 17, n) + 0.3 * (edad - 63), 90, 220).round(0)
    tab = np.where(r.random(n) < 0.12, 'Actual', np.where(r.random(n) < 0.33, 'Exfumador', 'Nunca'))
    ieca = r.random(n) < 0.72
    lp = (-0.050 * (tfg - 38) + 0.0030 * np.clip(25 - tfg, 0, None) ** 2 + 0.42 * np.log(uacr / 300) + 0.30 * dm - 0.14 * (hb - 12) - 0.07 * (bic - 24) + 0.33 * (fos - 3.7) - 0.012 * (edad - 63) + 0.15 * (tab == 'Actual') - 0.2 * ieca * 0)
    lp = lp * (0.85 if shift else 1.0) + (0.35 if shift else 0.0)
    shape = 1.3
    # tiempo Weibull con riesgos proporcionales; la constante (intercepto) se calibra para un riesgo medio a 2 años ≈ 16 % en desarrollo
    if 'dev' not in B0:
        lo, hi = -12.0, 2.0
        for _ in range(60):
            mid = (lo + hi) / 2; risk = np.mean(1 - np.exp(-np.exp(mid + lp) * 2 ** shape))
            lo, hi = (mid, hi) if risk < 0.16 else (lo, mid)
        B0['dev'] = (lo + hi) / 2
    u = r.random(n); b0 = B0['dev']
    t = (-np.log(u) / (np.exp(b0 + lp))) ** (1 / shape)
    cens = r.uniform(1.8, 6.0, n) if not shift else r.uniform(1.6, 5.0, n)
    time = np.minimum(t, cens); ev = (t <= cens)
    df = pd.DataFrame({'ID': np.arange(1, n + 1), 'Edad (años)': edad, 'Sexo': np.where(sexo, 'Masculino', 'Femenino'), 'Diabetes': np.where(dm, 'Sí', 'No'), 'Hipertensión': np.where(hta, 'Sí', 'No'),
        'TFGe (mL/min/1.73 m²)': tfg, 'UACR (mg/g)': uacr, 'Hemoglobina (g/dL)': hb, 'Bicarbonato (mmol/L)': bic, 'Fosfato (mg/dL)': fos, 'Albúmina (g/dL)': alb, 'PAS (mmHg)': pas,
        'Tabaquismo': tab, 'IECA/ARA-II': np.where(ieca, 'Sí', 'No'), 'Seguimiento (años)': time.round(2), 'Falla renal': np.where(ev, 'Sí', 'No')})
    # desenlace binario a 2 años: evento antes de 2 años = Sí; seguimiento completo ≥ 2 años sin evento = No; censurado antes de 2 años = perdido
    b = np.where(ev & (time <= 2), 'Sí', np.where(time >= 2, 'No', ''))
    df['Falla renal a 2 años'] = b
    # datos perdidos
    for col, p in [('Hemoglobina (g/dL)', 0.05), ('UACR (mg/g)', 0.07), ('Fosfato (mg/dL)', 0.09)]:
        m = r.random(n) < p * np.where(dm, 1.6, 0.6); df.loc[m, col] = np.nan
    return df
simulate(1800, 101).to_csv('examples/erc_prediccion_desarrollo.csv', index=False)
simulate(900, 202, shift=True).to_csv('examples/erc_prediccion_validacion.csv', index=False)
d = pd.read_csv('examples/erc_prediccion_desarrollo.csv'); print(d.shape, (d['Falla renal'] == 'Sí').mean(), (d['Falla renal a 2 años'] == 'Sí').sum(), d['Falla renal a 2 años'].isna().sum())
v = pd.read_csv('examples/erc_prediccion_validacion.csv'); print(v.shape, (v['Falla renal'] == 'Sí').mean(), (v['Falla renal a 2 años'] == 'Sí').sum())
