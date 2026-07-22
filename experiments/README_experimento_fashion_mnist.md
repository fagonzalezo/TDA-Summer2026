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

## 6. ¿Cambia el panorama si a UMAP se le da la misma dimension que EmbedKit sugiere?

Las secciones 4 y 5 comparan a UMAP en 2D/3D contra EmbedKit tambien en
2D/3D — pero `EmbedKitAnalyzer` (Seccion 3) sugirio `target_dim=16` como la
dimension "comoda" para la geometria de estos datos. Para que la
comparacion no favorezca a EmbedKit por default (al dejarlo en su
dimension sugerida mientras UMAP se evalua en 2D/3D), se repitio **todo el
experimento dandole tambien 16 dimensiones a UMAP** (y a PCA, como
referencia adicional), tanto en la Seccion A como en la prueba de
generalizacion (`DIM_GEN=16`).

**Seccion A a 16D — metricas de una sola pasada:**

| Metodo | Trustworthiness | Continuity | kNN acc | Silhouette | Corr. distancias | Tiempo (s) |
|---|---|---|---|---|---|---|
| PCA (16D) | **0.995** | **0.998** | 0.768 | 0.076 | **0.981** | 0.06 |
| UMAP (16D) | 0.985 | 0.986 | 0.737 | 0.178 | 0.612 | 13.0 |
| EmbedKit self-sup (16D) | 0.994 | 0.997 | **0.782** | 0.048 | 0.297 | 17.7 |
| EmbedKit superv. (16D) *(referencia, usa `y`)* | 0.952 | 0.968 | 0.941 | 0.449 | 0.544 | 16.1 |

(CSV completo, con las filas 2D/3D originales incluidas:
`resultados_fashion_mnist_dim16/tabla_metricas_2d.csv`)

**A 16D, la brecha de la Seccion A se cierra e incluso se invierte en las
metricas locales:** EmbedKit self-sup ahora **empata o supera** a UMAP en
Trustworthiness (0.994 vs 0.985), Continuity (0.997 vs 0.986) y kNN acc
(0.782 vs 0.737). Dandole a EmbedKit el espacio que su propio diagnostico
pide, deja de colapsar (ya no hay anillo *S*¹) y preserva vecindarios
locales muy bien. El costo es la **estructura global**: su correlacion de
distancias cae a 0.297 — la peor de las cuatro, incluso peor que en 2D/3D
(0.44–0.59) — es decir, al usar mas dimensiones para ajustar vecindarios
locales, distorsiona aun mas las distancias entre grupos lejanos. Tambien
resalta que **PCA a 16D es sorprendentemente fuerte en casi todo**
(Trustworthiness, Continuity y Corr. distancias mas altas que las de
cualquier metodo no lineal): con 16 de 784 dimensiones, la proyeccion
lineal ya captura la mayor parte de la varianza util en este dataset.

**Generalizacion fuera de muestra a `DIM_GEN=16`:**

| Metodo | OOS kNN acc (media ± std) | OOS Trustworthiness |
|---|---|---|
| PCA | 0.772 ± 0.011 | 0.993 |
| **UMAP** | **0.734 ± 0.011** | 0.975 |
| **EmbedKit self-sup** | **0.588 ± 0.151** | 0.792 |
| EmbedKit superv. *(referencia, usa `y`)* | 0.817 ± 0.010 | 0.909 |
| Base (R⁷⁸⁴, sin embeber) | 0.774 ± 0.016 | — |

(CSV: `resultados_fashion_mnist_dim16/generalizacion_oos.csv`, figura:
`resultados_fashion_mnist_dim16/fig_generalizacion.png`)

**Aqui es donde el hallazgo central se mantiene, aunque atenuado.** Con
16 dimensiones, EmbedKit self-sup mejora bastante frente a `DIM_GEN=3`
(media 0.345→0.588, std 0.235→0.151), pero **sigue siendo ~14 veces mas
inestable que UMAP** (std 0.151 vs 0.011) y su media sigue por debajo de
PCA, UMAP y hasta del espacio crudo sin embeber. Es decir: **darle a
EmbedKit self-sup su dimension "comoda" ayuda, pero no resuelve su
inestabilidad al reentrenarse en cada fold** — el problema no era solo de
dimension insuficiente. UMAP, en cambio, es virtualmente igual de estable
en 3D y en 16D (std ≈ 0.009–0.011 en ambos casos).

Un dato adicional notable: **EmbedKit superv. a 16D supera a UMAP en
generalizacion** (0.817 vs 0.734) y con muy baja variabilidad (std 0.010,
mucho menor que su propio 0.109 a `DIM_GEN=3`) — con etiquetas y
suficiente dimension, el refinador contrastivo se vuelve competitivo y
estable. Pero esto usa `y` para entrenar, por lo que sigue siendo una
referencia de cota superior y no cambia la comparacion **no supervisada**.

**Config de esta corrida:** `N=3000`, `SEED=0`, `K=10`, `epochs=150`,
`DIM_GEN=16`, `--dim-alt 16` (ver `resultados_fashion_mnist_dim16/config.json`).

## 7. Conclusiones

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
   muestra mas variabilidad entre folds que UMAP a `DIM_GEN=3` (aunque esto
   se revierte a `DIM_GEN=16`, ver punto 6).
6. **Darle a UMAP y a EmbedKit la dimension que el propio `EmbedKitAnalyzer`
   sugiere (16) matiza, pero no revierte, el hallazgo central.** En
   metricas de una sola pasada (Seccion A), EmbedKit self-sup deja de
   colapsar y hasta **empata o supera a UMAP** en Trustworthiness,
   Continuity y kNN acc — a costa de una correlacion de distancias mucho
   peor (0.297, la mas baja de todas). Pero en la prueba que de verdad
   importa — **generalizacion fuera de muestra** — EmbedKit self-sup sigue
   siendo **~14 veces mas inestable que UMAP** (std 0.151 vs 0.011) aunque
   su media mejore; UMAP, en cambio, es igual de estable en 3D que en 16D.
   Es decir, la dimension insuficiente no era la unica causa de la
   inestabilidad de EmbedKit self-sup. Ver Seccion 6 para el detalle.

**En resumen:** en Fashion-MNIST, un dataset con clases mas dificiles de
separar y una geometria de embedding cruda mas anisotropica que `digits`,
**UMAP se mantiene como el *embedder* no supervisado mas confiable**,
sobre todo en **estabilidad al generalizar** a datos nuevos — una ventaja
que persiste incluso cuando a EmbedKit se le da la dimension que su propio
diagnostico recomienda. En calidad de proyeccion de una sola pasada
(Seccion A), sin embargo, la ventaja de UMAP sobre EmbedKit self-sup
**depende de la dimension**: es clara en 2D/3D y se **cierra casi por
completo (o se invierte en metricas locales) a 16D**, aunque a costa de
que EmbedKit sacrifique la estructura global de las distancias. El
refinador no supervisado de `embedding-kit` no cierra la brecha de
**estabilidad**, que sigue siendo su debilidad mas pronunciada frente a
UMAP en este dataset mas complejo.

## 8. Como reproducir

```bash
cd experiments
pip install -r requirements.txt

# Comparacion original (2D/3D, mas la pasada extra a dim=16 en Seccion A)
python fashion_mnist_umap_vs_embedkit.py --n 3000 --outdir resultados_fashion_mnist

# Generalizacion tambien a dim=16 para UMAP y EmbedKit (Seccion 6)
python fashion_mnist_umap_vs_embedkit.py --n 3000 --dim-gen 16 --outdir resultados_fashion_mnist_dim16
```

Salidas generadas en cada `--outdir`:
`tabla_metricas_2d.csv`, `generalizacion_oos.csv`, `config.json`,
`fig_embeddings_2d.png`, `fig_metricas_barras.png`, `fig_generalizacion.png`.

Notas:
- Si `sklearn.datasets.fetch_openml` no puede descargar Fashion-MNIST (p. ej.
  por politicas de red), el script cae automaticamente a
  `torchvision.datasets.FashionMNIST`.
- `--con-tsne` agrega t-SNE a la Seccion A (mas lento). `--sin-supervisado`
  omite la fila de referencia de EmbedKit supervisado. `--dim-alt` controla
  la dimension adicional de la Seccion A (por defecto 16, la sugerida por
  `EmbedKitAnalyzer`; `0` la omite). `--dim-gen` controla la dimension de la
  prueba de generalizacion (Seccion B/6, por defecto 3).
