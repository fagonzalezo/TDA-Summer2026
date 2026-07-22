# Experimento: UMAP vs. EmbedKit (no supervisado) en Fashion-MNIST

## 1. Motivacion

En el notebook del curso `UNAL_Bog_TDA_Comparacion_Reduccion_Dimensionalidad.ipynb`
se compararon varias tecnicas de reduccion de dimensionalidad sobre `load_digits()`
(64 dimensiones, 10 clases). Una de las observaciones de ese notebook es que
**UMAP se comporta muy bien como *embedder* no supervisado**, mientras que el
refinador contrastivo de [`embedding-kit`](https://github.com/fagonzalezo/embedding-kit)
en su modo **no supervisado** (`self_supervised`) no domina en las metricas
no supervisadas y, sobre todo, **se vuelve inestable** al evaluar su capacidad
de generalizar a datos nuevos.

Este experimento pone esa observacion a prueba en un dataset **mas complejo**:
**Fashion-MNIST** (imagenes de ropa, 28×28 = 784 dimensiones, 10 clases con
bastante solapamiento visual, p. ej. camisa/abrigo/saco). El objetivo es
responder: **¿se mantiene, se atenua o se agrava la ventaja de UMAP sobre el
refinamiento no supervisado de EmbedKit cuando el dataset es mas dificil y de
mayor dimension?**

Este documento reporta los resultados **reales** obtenidos al correr
`fashion_mnist_umap_vs_embedkit.py` (no son numeros ilustrativos).

## 2. Metodologia

Se replica el diseño experimental del notebook de `digits`, cambiando solo el
dataset, para que la comparacion entre ambos experimentos sea directa:

- **Dataset:** Fashion-MNIST completo (70 000 imagenes; se usaron 60 000 del
  split de entrenamiento), pixeles escalados a `[0, 1]`.
- **Submuestra estratificada:** `N = 3000` (300 puntos por clase), semilla
  `SEED = 0`, mismo patron de muestreo del notebook (`rng.choice` por clase).
- **Vecindad comun:** `K = 10` para todos los metodos que la usan
  (`n_neighbors`, kNN de evaluacion, trustworthiness/continuity).
- **Metricas** (funcion `evaluar`, identica a la del notebook):
  1. **Trustworthiness** (no sup.) — ¿los vecinos en baja dimension eran
     realmente vecinos en alta dimension?
  2. **Continuity** (no sup.) — la inversa: ¿los vecinos en alta dimension se
     preservan como vecinos en baja dimension?
  3. **kNN acc** (**sup.**, usa `y`) — exactitud de un k-NN (5-fold CV) sobre
     el embedding.
  4. **Silhouette** (**sup.**, usa `y`) — separacion de los grupos por clase.
  5. **Corr. distancias** (no sup., Spearman) — preservacion de la estructura
     **global** de distancias por pares.
  + tiempo de computo.
- **Generalizacion fuera de muestra** (replica la Seccion 8 del notebook):
  `StratifiedKFold` de 5 particiones; cada encoder se ajusta **solo** con el
  train de cada fold, se proyecta el test, y se mide la exactitud de un k-NN
  entrenado sobre el train embebido y evaluado en el test embebido — la
  prueba clave de si el metodo **generaliza** o solo memoriza la muestra.
  Dimension de embedding para esta seccion: `DIM_GEN = 3` (misma para todos).

**Metodos comparados:**

| Metodo | Tipo | Usa etiquetas |
|---|---|---|
| PCA | lineal, baseline | No |
| **UMAP** | no lineal, protagonista no supervisado | No |
| **EmbedKit self-sup (2D)** | refinador contrastivo no supervisado | No |
| EmbedKit self-sup (3D) | idem, en 3D (esfera *S*²) | No |
| EmbedKit superv. (3D) | refinador contrastivo supervisado (SupCon) | **Si** (cota superior de referencia) |

> Las filas de EmbedKit en 3D miden sobre la esfera *S*² — tienen una ventaja
> estructural leve frente a las de 2D (mas espacio para preservar vecindarios).
> La fila **supervisada** usa las etiquetas al entrenar el encoder y luego se
> evalua con metricas que tambien usan etiquetas (kNN acc, Silhouette): sus
> valores estan **inflados** por diseño y se incluye solo como cota superior
> de referencia, no como parte de la comparacion no supervisada.

**Configuracion de la corrida:** `N=3000`, `SEED=0`, `K=10`, `epochs=150`
(EmbedKit), `DIM_GEN=3`. Ver `resultados_fashion_mnist/config.json`.

## 3. Diagnostico previo de la geometria (EmbedKitAnalyzer)

Antes de refinar nada, `EmbedKitAnalyzer` diagnostico la geometria cruda de
los 3000 puntos de Fashion-MNIST en R⁷⁸⁴:

- **Dimension intrinseca (consenso):** 10.6 (±2.3) — muy baja frente a las
  784 dimensiones ambiente (ID/D ≈ 0.014).
- **`participation_ratio`:** 7.71, **`isotropy_score`:** 0.036 — geometria
  **muy anisotropica** (la energia se concentra en pocas direcciones).
- **Severidad:** `HIGH`, con recomendacion explicita de reduccion fuerte de
  dimensionalidad y de considerar *whitening* por la baja isotropia.
- `suggested_target_dim`: 16.

Este diagnostico ya anticipa un reto: Fashion-MNIST tiene una geometria de
embedding cruda **mas anisotropica y con mayor severidad** que la de
`digits` en el notebook original (alli el analizador sugeria `target_dim≈12`
con menor severidad reportada), lo que es consistente con que sea un dataset
mas dificil para un refinador aprendido con pocos datos.

## 4. Resultados — Seccion A: embeddings 2D/3D y metricas

| Metodo | Trustworthiness | Continuity | kNN acc | Silhouette | Corr. distancias | Tiempo (s) |
|---|---|---|---|---|---|---|
| PCA | 0.917 | 0.972 | 0.525 | -0.030 | **0.880** | 0.04 |
| **UMAP** | **0.980** | **0.983** | 0.741 | 0.177 | 0.625 | 28.6 |
| EmbedKit self-sup (2D) | 0.805 | 0.953 | 0.491 | -0.027 | 0.444 | 18.7 |
| EmbedKit self-sup (3D) | 0.954 | 0.986 | 0.651 | 0.056 | 0.588 | 24.3 |
| EmbedKit superv. (3D) *(referencia, usa `y`)* | 0.840 | 0.940 | 0.931 | 0.601 | 0.512 | 25.3 |

(CSV completo: `resultados_fashion_mnist/tabla_metricas_2d.csv`)

**Comparacion directa a la misma dimension (2D), no supervisada:**
UMAP **domina en las tres metricas no supervisadas** frente a EmbedKit
self-sup: Trustworthiness (0.980 vs 0.805), Continuity (0.983 vs 0.953) y,
sobre todo, en las metricas que usan etiquetas para *evaluar* (aunque no
para entrenar) — kNN acc (0.741 vs 0.491) y Silhouette (0.177 vs -0.027).

El caso mas llamativo es visual: en la figura `fig_embeddings_2d.png`,
**EmbedKit self-sup (2D) colapsa a un anillo** (proyeccion casi uniforme
sobre el circulo *S*¹) sin separacion visible entre clases — coherente con
su Silhouette practicamente nulo/negativo y su kNN acc apenas por encima del
azar. Subir a 3D (sobre la esfera *S*²) ayuda bastante a EmbedKit self-sup
(Trustworthiness sube a 0.954, kNN acc a 0.651), pero **sigue sin alcanzar a
UMAP en 2D** en kNN acc, y lo hace con una dimension extra de ventaja
estructural.

En cambio, **PCA gana claramente en correlacion de distancias** (0.880):
es la metrica que mejor recompensa preservar la estructura *global*, algo
que un metodo lineal hace naturalmente y que tanto UMAP como EmbedKit
sacrifican por preservar vecindarios *locales*.

## 5. Resultados — Seccion B: generalizacion fuera de muestra

Esta es la prueba mas dura y la que mas separa a los metodos: ¿el encoder
aprendido en el *train* de cada fold generaliza al *test* del mismo fold?

| Metodo | OOS kNN acc (media ± std) | OOS Trustworthiness |
|---|---|---|
| PCA | 0.620 ± 0.018 | 0.951 |
| **UMAP** | **0.726 ± 0.009** | **0.975** |
| **EmbedKit self-sup** | **0.345 ± 0.235** | 0.703 |
| EmbedKit superv. *(referencia, usa `y`)* | 0.680 ± 0.109 | 0.825 |
| Base (R⁷⁸⁴, sin embeber) | 0.774 ± 0.016 | — |

(CSV completo: `resultados_fashion_mnist/generalizacion_oos.csv`,
figura: `resultados_fashion_mnist/fig_generalizacion.png`)

**Este es el resultado central del experimento.** UMAP generaliza de forma
**estable** (desviacion estandar de solo 0.009 entre los 5 folds) y con la
mejor exactitud fuera de muestra entre los metodos no supervisados (0.726).
EmbedKit self-sup, en cambio, **colapsa en algunos folds**: su desviacion
estandar (0.235) es **26 veces mayor** que la de UMAP, y su media (0.345)
queda muy por debajo — incluso de PCA. Esto **confirma y agrava** la
inestabilidad que el notebook original ya señalaba para `digits` con `N`
bajo: aqui, con `N=3000` (el doble de la muestra "estable" que usaba el
notebook para `digits`) y sobre un dataset mas dificil, **EmbedKit
self-sup sigue siendo inestable**, mientras que UMAP no muestra ese
problema.

Notese ademas que **incluso el modo supervisado de EmbedKit** (que usa
etiquetas) muestra una desviacion estandar (0.109) mucho mayor que la de
UMAP, sugiriendo que la inestabilidad del refinador contrastivo entrenado
con pocos datos por fold (2400 puntos de train) no es exclusiva del modo no
supervisado, aunque es notablemente peor sin etiquetas.

## 6. Conclusiones

1. **La observacion del notebook original se confirma y se acentua en un
   dataset mas complejo.** En Fashion-MNIST, UMAP supera a EmbedKit
   self-sup en las tres metricas no supervisadas de la Seccion A
   (trustworthiness, continuity) y, sobre todo, en la prueba de
   generalizacion fuera de muestra, donde la diferencia es dramatica.
2. **La inestabilidad de EmbedKit self-sup no es un artefacto de muestras
   pequeñas.** El notebook de `digits` la atribuia a `N` bajo (300–500
   puntos). Aqui, con `N=3000` sobre un dataset mas dificil, el refinador
   no supervisado **sigue colapsando** en algunos folds (2400 puntos de
   entrenamiento por fold) — la dificultad intrinseca del dataset (menor
   isotropia, mayor severidad segun `EmbedKitAnalyzer`) parece pesar tanto
   o mas que el tamaño de muestra.
3. **El colapso a un anillo en 2D** (EmbedKit self-sup) es el sintoma mas
   visible: sin supervision, el proyector contrastivo no encuentra una
   señal suficiente en 784 dimensiones tan anisotropicas para separar 10
   clases visualmente similares en solo 2 dimensiones.
4. **PCA sigue siendo el mejor en preservar estructura global** (correlacion
   de distancias), recordando que ninguna tecnica domina en todas las
   metricas — el objetivo de la evaluacion determina el ganador.
5. **La cota supervisada de EmbedKit** logra la mejor separacion de clases
   en la Seccion A (Silhouette 0.601, kNN acc 0.931) pero, al usar
   etiquetas tanto para entrenar como para evaluar, estos valores estan
   inflados por diseño; en generalizacion, incluso el modo supervisado
   muestra mas variabilidad entre folds que UMAP.

**En resumen:** en Fashion-MNIST, un dataset con clases mas dificiles de
separar y una geometria de embedding cruda mas anisotropica que `digits`,
**UMAP se mantiene como el mejor *embedder* no supervisado**, tanto en
calidad de proyeccion como — sobre todo — en **estabilidad al generalizar**
a datos nuevos. El refinador no supervisado de `embedding-kit` no solo no
cierra esa brecha en un dataset mas complejo, sino que su principal
debilidad (inestabilidad fuera de muestra) se hace **mas pronunciada**.

## 7. Como reproducir

```bash
cd experiments
pip install -r requirements.txt
python fashion_mnist_umap_vs_embedkit.py --n 3000
```

Salidas generadas en `resultados_fashion_mnist/`:
`tabla_metricas_2d.csv`, `generalizacion_oos.csv`, `config.json`,
`fig_embeddings_2d.png`, `fig_metricas_barras.png`, `fig_generalizacion.png`.

Notas:
- Si `sklearn.datasets.fetch_openml` no puede descargar Fashion-MNIST (p. ej.
  por politicas de red), el script cae automaticamente a
  `torchvision.datasets.FashionMNIST`.
- `--con-tsne` agrega t-SNE a la Seccion A (mas lento). `--sin-supervisado`
  omite la fila de referencia de EmbedKit supervisado.
