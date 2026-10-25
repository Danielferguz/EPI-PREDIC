export const HELP_HTML = `
<h2>Cómo usar EPI-PREDIC</h2>
<p>EPI-PREDIC guía el <b>desarrollo y la validación de modelos de predicción clínica</b> (diagnósticos y pronósticos) siguiendo la serie de Collins, Riley y colaboradores en el <i>BMJ</i> y la guía TRIPOD+AI. Todo se calcula en su navegador.</p>
<h3>Flujo recomendado</h3>
<ol>
<li><b>Tamaño de muestra</b> (antes de recoger datos): criterios de Riley et al. para desarrollo (binario, supervivencia, continuo) y para validación externa (binario y continuo).</li>
<li><b>Datos</b>: una fila por paciente; desenlace y predictores medidos antes del momento de predicción.</li>
<li><b>Modelo</b>: logístico, Cox o lineal; formas funcionales (lineal, logaritmo, splines cúbicos restringidos de 3–5 nudos); penalización ridge/lasso (λ por validación cruzada); número de réplicas bootstrap.</li>
<li><b>Desarrollo</b>: coeficientes, rendimiento aparente y <i>corregido por optimismo</i> (bootstrap de Harrell que repite todo el proceso de modelado), factor de contracción uniforme y curva de calibración corregida.</li>
<li><b>Validación</b> en otra base (otro centro/periodo) o de un <i>modelo publicado</i> (ecuación): C, pendiente de calibración, CITL, O/E, calibración y suficiencia de la muestra. Recalibración del intercepto y de la pendiente.</li>
<li><b>Utilidad clínica</b>: curva de decisión (beneficio neto) para desenlaces binarios y de supervivencia (con censura).</li>
<li><b>Reporte</b>: texto de métodos (ES/EN), tablas Word/CSV, ecuación del modelo, calculadora de riesgo y lista TRIPOD+AI.</li></ol>
<p><b>Con EPI-Maching-Learning.</b> EPI-PREDIC <i>desarrolla y valida</i> un modelo de regresión (tamaño de muestra, contracción por bootstrap, validación externa, TRIPOD+AI); <a href="../EPI-Maching-Learning/" target="_blank" rel="noopener">EPI-Maching-Learning</a> <i>compara algoritmos</i> flexibles y el Super Learner, orienta qué modelos son razonables según los eventos, simplifica el modelo por importancia por permutación y descarga código Python y una calculadora de riesgo. Flujo sugerido: tamaño de muestra aquí → comparar algoritmos y simplificar allí → modelo final y validación externa aquí.</p>
<h3>Qué mide cada indicador</h3>
<table class="data"><thead><tr><th>Indicador</th><th>Qué responde</th><th>Valor ideal</th></tr></thead><tbody>
<tr><td>Estadístico C (AUC) / índice C de Harrell</td><td>¿Ordena bien a los pacientes con y sin evento? (discriminación)</td><td>1 (0.5 = azar)</td></tr>
<tr><td>Pendiente de calibración</td><td>¿Son los riesgos demasiado extremos (&lt; 1) o demasiado tímidos (&gt; 1)?</td><td>1</td></tr>
<tr><td>CITL / O/E</td><td>¿Se acierta el riesgo medio? Intercepto de calibración con pendiente fija / observados entre esperados</td><td>0 / 1</td></tr>
<tr><td>Curva de calibración</td><td>¿Coincide el riesgo predicho con el observado en todo el rango?</td><td>Diagonal</td></tr>
<tr><td>Brier</td><td>Error cuadrático medio de las probabilidades (mezcla discriminación y calibración)</td><td>Menor es mejor</td></tr>
<tr><td>R² de Cox-Snell</td><td>Variación explicada (generalizada); alimenta el cálculo del tamaño de muestra</td><td>Mayor es mejor</td></tr>
<tr><td>Beneficio neto</td><td>¿Mejora decisiones frente a tratar a todos o a nadie, dado un umbral de riesgo?</td><td>&gt; 0 y &gt; «tratar a todos»</td></tr></tbody></table>
<h3>Tamaño de muestra: criterios implementados</h3>
<ul><li><b>Desarrollo, binario y supervivencia.</b> (i) factor de contracción global S ≥ 0.9: <code>n = p / ((S − 1)·ln(1 − R²<sub>CS</sub>/S))</code>; (ii) diferencia ≤ 0.05 entre el R² de Nagelkerke aparente y el ajustado; (iii) margen de error ≤ 0.05 en el riesgo global (binario: <code>(1.96/0.05)²·φ(1−φ)</code>; supervivencia: IC del riesgo acumulado). El R² de Cox-Snell puede obtenerse del R² de Nagelkerke o del estadístico C (integración numérica del modelo binormal).</li>
<li><b>Desarrollo, continuo.</b> contracción global, diferencia de R², precisión de la varianza residual (n = 234 + p) y precisión del intercepto (MMOE ≤ 1.1).</li>
<li><b>Validación externa, binario.</b> precisión de O/E, de la pendiente de calibración (información de Fisher del modelo de calibración con la distribución del predictor lineal), del C (error estándar de Newcombe) y del beneficio neto estandarizado (Marsh et al.). <b>Continuo:</b> R², CITL, pendiente y varianza residual (≥ 235).</li></ul>
<p>Los cálculos de desarrollo reproducen al paquete <code>pmsampsize</code> en 53 escenarios; los de validación se comprobaron con los ejemplos del <i>BMJ</i> (886/184/258/235) y por simulación de Monte Carlo.</p>
<h3>Advertencias</h3>
<ul><li>La selección de variables por valor p o por pasos <b>no</b> se ofrece: se desaconseja en modelos de predicción. Elija los predictores por conocimiento clínico y use penalización si hay pocos eventos.</li>
<li>Con datos perdidos, el análisis de casos completos puede sesgar. Impute con EPI-DATA (MICE) y cargue la base imputada, o reduzca los predictores con muchos perdidos.</li>
<li>La validación interna corrige el sobreajuste, <b>no</b> sustituye a la validación externa. La partición aleatoria en entrenamiento/prueba es ineficiente y no se ofrece.</li>
<li>Un modelo predictivo no estima efectos causales: no interprete sus coeficientes como tal (ver EPI-CAUSAL).</li></ul>
<h3>Referencias</h3>
<ul><li>Collins GS et al. Evaluation of clinical prediction models (part 1). <i>BMJ</i> 2024;384:e074819. · Riley RD et al. (part 2) e074820; (part 3: tamaño de muestra de validación) e074821.</li>
<li>Riley RD et al. Calculating the sample size required for developing a clinical prediction model. <i>BMJ</i> 2020;368:m441. · Riley RD et al. <i>Stat Med</i> 2019;38:1276–96 y 1297–306.</li>
<li>Collins GS et al. TRIPOD+AI statement. <i>BMJ</i> 2024;385:e078378. · Efthimiou O et al. Developing clinical prediction models: a step-by-step guide. <i>BMJ</i> 2024;386:e078276.</li>
<li>Steyerberg EW. <i>Clinical Prediction Models</i>, 2.ª ed. Springer, 2019. · Harrell FE. <i>Regression Modeling Strategies</i>, 2.ª ed. Springer, 2015. · Vickers AJ, Elkin EB. <i>Med Decis Making</i> 2006;26:565–74.</li></ul>
<h3>Licencia</h3><p>© Daniel Fernández-Guzmán. CC BY-NC-ND 4.0. Usar la aplicación con sus datos y publicar los resultados está permitido citando EPI-PREDIC.</p>`;
