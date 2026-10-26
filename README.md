# EPI-PREDIC

Módulo de **EPI-LAB** para **desarrollar, validar y reportar modelos de predicción clínica** (diagnósticos y pronósticos) siguiendo la serie del *BMJ* de Collins, Riley y colaboradores (*Evaluation of clinical prediction models*, partes 1–3), la guía paso a paso de Efthimiou et al. y TRIPOD+AI. Incluye el **cálculo del tamaño de muestra**.

Todo se calcula en el navegador: los datos no salen de tu computadora. Sitio estático (GitHub Pages / Cloudflare Pages), sin instalación.

## Funciones

| Paso | Contenido |
|---|---|
| 1 · Tamaño de muestra | **Desarrollo** (binario, supervivencia, continuo): criterios de Riley et al. (contracción global, diferencia de R², precisión del riesgo/intercepto/varianza), R² de Cox-Snell desde Nagelkerke o desde el estadístico C, sensibilidad y texto para métodos (ES/EN). **Validación externa** (binario y continuo): O/E, pendiente de calibración, C, beneficio neto estandarizado, R², CITL |
| 2 · Datos | CSV/Excel; tipos de variable; ejemplos simulados de nefrología (desarrollo y validación) |
| 3 · Modelo | Logístico, Cox o lineal; forma lineal, logarítmica o spline cúbico restringido (3–5 nudos, Harrell); ridge/lasso (λ por CV, regla mínima o 1-ES); selección hacia atrás por AIC, BIC o QIC repetida en cada réplica bootstrap (con frecuencia de inclusión); índice C de Harrell también para desenlaces continuos; comparación de modelos con el mismo bootstrap; comprobación de EPP y de la muestra frente al paso 1 |
| 4 · Desarrollo | Coeficientes (OR/HR con IC, pruebas globales y de no linealidad), rendimiento aparente y **corregido por bootstrap de Harrell** (repite todo el proceso, incluida la elección de λ), factor de contracción uniforme, calibración corregida (lowess), ROC, ecuación y descarga del modelo |
| 5 · Validación | Aplicación del modelo en otra base (C con IC, pendiente, CITL, O/E, calibración, comparación de poblaciones, suficiencia de la muestra), **validación de un modelo publicado** (ecuación) y recalibración |
| 6 · Utilidad clínica | Curva de decisión (beneficio neto, intervenciones evitadas) para desenlaces binarios y de supervivencia con censura |
| 4 · Desarrollo (adicional) | **Tabla 1** (total y por desenlace, con p), **regresión cruda y ajustada** por predictor, gráficos de rendimiento (distribución de riesgos, Se/Sp/VPP/VPN por umbral, riesgo por grupos) y **simplificación del modelo** sin perder capacidad de predicción (bootstrap que repite el orden de importancia) |
| 7 · Reporte | Texto de métodos (ES/EN), tablas Word/CSV, ecuación, calculadora de riesgo (también descargable como HTML autónomo), **código Python sin figuras**, **borrador de manuscrito ES/EN** (con tablas y figuras) y lista TRIPOD+AI |

## Validación de los motores

```bash
node tests/validate_samplesize.mjs   # frente a pmsampsize (Python), ejemplos publicados y Monte Carlo
node tests/validate.mjs              # frente a statsmodels, scikit-learn y lifelines; corrección del optimismo por simulación
python3 tools/make_reference.py      # regenera las referencias (requiere pmsampsize, scikit-learn, statsmodels, lifelines)
```

* Tamaño de muestra de desarrollo: 53 escenarios idénticos a `pmsampsize`; ejemplos de las diapositivas de Riley; R² de Cox-Snell desde C (integración numérica) frente a la simulación del paquete.
* Validación externa: ejemplo del BMJ (886 / 184 / 258 / 235) y simulaciones de Monte Carlo que confirman que el error estándar empírico coincide con el objetivo.
* Regresión logística/Cox/penalizada: coeficientes, AUC, Brier, calibración, C de Harrell con empates y riesgos de Breslow frente a statsmodels, scikit-learn y lifelines; lasso/ridge coinciden con scikit-learn a 1e-9; la selección hacia atrás por AIC/BIC/QIC coincide con statsmodels.
* Bootstrap: en 30 réplicas simuladas la pendiente y el C corregidos coinciden con los valores «verdaderos» en una muestra independiente de 20 000.

## Decisiones de diseño

* **No** se ofrece selección de variables por valor p ni por pasos (desaconsejada en predicción) ni la partición simple entrenamiento/prueba; sí penalización y contracción.
* Los datos perdidos se manejan por casos completos; para imputación use EPI-DATA (MICE) y cargue la base imputada.
* Los coeficientes de un modelo predictivo no son efectos causales (véase EPI-CAUSAL).

## Uso local

```bash
python3 -m http.server 8000   # luego http://localhost:8000
```

## Datos de ejemplo (simulados)

`examples/erc_prediccion_desarrollo.csv` (1800 pacientes) y `erc_prediccion_validacion.csv` (900, otro «centro» con más riesgo y efectos más débiles; se generan con `tools/make_examples.py`), más los ejemplos de nefrología de EPI-DATA.

## Licencia y cita

© Daniel Fernández-Guzmán. Licencia **CC BY-NC-ND 4.0**. Usar la aplicación y publicar los resultados está permitido citando EPI-PREDIC (ver `CITATION.cff`).
