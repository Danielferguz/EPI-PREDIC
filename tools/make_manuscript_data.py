# Datos SIMULADOS para el ejemplo de manuscrito: cohorte de ERC (todas las etiologías) con terapia de reemplazo renal (TRR) a 5 años.
# Inspirado en la estructura de un estudio de validación del score KFRE por etiología; ningún dato es real.
import numpy as np, pandas as pd
rng = np.random.default_rng(2026); n = 2400
etio = rng.choice(['Vascular', 'Diabetic', 'Glomerular', 'Tubulointerstitial', 'Polycystic', 'Unclassified'], n, p=[.37, .21, .16, .14, .04, .08])
age = np.clip(rng.normal(70, 13, n) - 17 * (etio == 'Polycystic') - 8 * (etio == 'Glomerular'), 19, 98).round()
sex = np.where(rng.random(n) < .60, 'Male', 'Female')
diab = np.where((etio == 'Diabetic') | (rng.random(n) < .12), 'Yes', 'No')
htn = np.where(rng.random(n) < np.where(etio == 'Vascular', .95, .70), 'Yes', 'No')
egfr = np.clip(rng.lognormal(np.log(42), .38, n) + 8 * (etio == 'Polycystic'), 6, 110).round(1)
uacr = np.exp(rng.normal(np.log(120), 1.6, n) + 1.1 * (etio == 'Glomerular') + .6 * (etio == 'Diabetic') - .5 * (etio == 'Polycystic')).round(1)
hb = np.clip(rng.normal(12.6, 1.7, n) - .035 * (60 - egfr) / 5, 6.5, 17).round(1)
alb = np.clip(rng.normal(4.1, .45, n) - .25 * (np.log(uacr) > 6.2), 2.0, 5.2).round(2)
lp0 = - 0.052 * (egfr - 40) / 1 + 0.40 * np.log(uacr / 100) - 0.012 * (age - 70) + 0.15 * (sex == 'Male') + 0.25 * (diab == 'Yes') - 0.12 * (hb - 12.5) - 0.5 * (alb - 4.1) + 0.9 * (etio == 'Polycystic') + 0.15 * (etio == 'Glomerular')
w = rng.weibull(1.25, n); cens = np.minimum(5.0, rng.uniform(1.2, 9.0, n))
def rate(b0): t = np.exp(-(b0 + lp0) / 1.25) * w; return np.mean(t <= cens)
lo, hi = -15.0, 5.0
for _ in range(60):  # intercepto para que la TRR observada a 5 años sea ≈ 13 %
    mid = (lo + hi) / 2; lo, hi = (mid, hi) if rate(mid) < 0.13 else (lo, mid)
lp = lo + lp0; t = np.exp(-lp / 1.25) * w
time = np.minimum(t, cens).round(2); event = (t <= cens).astype(int) & (time <= 5.0)
time = np.minimum(time, 5.0); krt5 = np.where(event & (time <= 5.0), 'Yes', 'No')
uacr_s = uacr.astype(object); uacr_s[rng.random(n) < .12] = np.nan   # 12 % de albuminuria sin medir
df = pd.DataFrame({'Age (years)': age.astype(int), 'Sex': sex, 'Diabetes': diab, 'Hypertension': htn, 'CKD etiology': etio, 'eGFR (mL/min/1.73 m2)': egfr, 'Albuminuria, UACR (mg/g)': uacr_s,
                   'Haemoglobin (g/dL)': hb, 'Serum albumin (g/dL)': alb, 'Follow-up (years)': time, 'Kidney replacement therapy': np.where(event, 'Yes', 'No'), 'KRT within 5 years': krt5})
df.to_csv('examples/erc_krt5_manuscrito.csv', index=False); print(df.shape, (df['KRT within 5 years'] == 'Yes').mean().round(3), df['Albuminuria, UACR (mg/g)'].isna().mean().round(3))
