# Experimento: reducción de dimensionalidad sobre EMBEDDINGS preentrenados de Fashion-MNIST

## 1. Motivación y qué cambia respecto al experimento original

El experimento `fashion_mnist_umap_vs_embedkit.py` (documentado en
`README_experimento_fashion_mnist.md`) comparó PCA, UMAP y el refinador
[`embedding-kit`](https://github.com/fagonzalezo/embedding-kit) sobre
Fashion-MNIST usando como entrada los **píxeles crudos** de cada imagen
(28×28 = 784 dimensiones en `[0, 1]`).

Este experimento **replica esa metodología cambiando una sola cosa**: en lugar
de alimentar a los modelos con píxeles crudos, primero se calcula un
**embedding de cada imagen con una red convolucional preentrenada en ImageNet**
(`tf.keras.applications`, por defecto **MobileNetV2**) y ese embedding
(R¹²⁸⁰) es la entrada de PCA / UMAP / EmbedKit y del k-NN de referencia.

> **La pregunta:** ¿mejora la calidad de las proyecciones y —sobre todo— la
> generalización fuera de muestra cuando los modelos parten de
> **representaciones semánticas** de una red preentrenada en lugar de píxeles
> crudos? ¿Se mantiene el hallazgo central del experimento anterior (UMAP
> estable vs. EmbedKit self-sup inestable) al cambiar la representación de
> entrada?

Todo lo demás es idéntico al experimento crudo: mismo `N=3000` (300 por clase),
misma semilla `SEED=0`, misma vecindad `K=10`, mismas métricas, mismas
secciones A y B, y **la misma submuestra de imágenes** (se verificó que el
k-NN base sobre los píxeles crudos de esta muestra da `0.774 ± 0.016`,
idéntico al del experimento original — es decir, la comparación es directa,
sobre exactamente las mismas 3000 imágenes).

## 2. Cómo se calculan los embeddings

El extractor toma la red preentrenada con `include_top=False, pooling='avg'`:
se le quita la cabeza de clasificación de ImageNet y su salida es el vector
tras el **global average pooling**, es decir, la representación que la red
aprendió, sin la capa final. Ese vector es el embedding de la imagen.

Pipeline por imagen (`extraer_embeddings_preentrenados.py`):

1. La imagen 28×28 en escala de grises se **replica a 3 canales** (RGB).
2. Se **reescala** a 224×224 (tamaño de entrada de ImageNet).
3. Se aplica el `preprocess_input` propio del modelo (MobileNetV2 lleva a
   `[-1, 1]`).
4. Se pasa por la red → vector R¹²⁸⁰ (MobileNetV2). Con `--modelo resnet50`
   serían R²⁰⁴⁸ y con `efficientnet_b0`, R¹²⁸⁰.

### Nota de implementación: dos procesos a propósito

TensorFlow (extractor) y PyTorch (que usa `embedding-kit`, y que `umap-learn`
importa de forma ansiosa en este entorno) cargados en el **mismo intérprete**
provocan un `Segmentation fault`. Por eso el experimento está partido en dos
pasos que **no comparten proceso**:

- **Paso 1 —** `extraer_embeddings_preentrenados.py`: sólo TensorFlow. Calcula
  los embeddings y los cachea en un `.npz`.
- **Paso 2 —** `fashion_mnist_embeddings_preentrenadas.py`: sólo
  PyTorch/UMAP/sklearn. Lanza el paso 1 como subproceso (intérprete aparte),
  lee el `.npz` y corre las secciones A y B. Además bloquea el módulo
  `tensorflow` antes de importar `umap` para que éste no lo cargue.

## 3. Diagnóstico de la geometría (EmbedKitAnalyzer): embeddings vs. píxeles

`EmbedKitAnalyzer` sobre los 3000 embeddings de MobileNetV2 (R¹²⁸⁰), comparado
con el diagnóstico sobre los píxeles crudos (R⁷⁸⁴) del experimento anterior:

| Métrica de geometría | Píxeles crudos (R⁷⁸⁴) | Embeddings (R¹²⁸⁰) |
|---|---|---|
| Dimensión intrínseca / ambiente (ID/D) | ≈ 0.014 | ≈ 0.01 |
| `participation_ratio` | 7.71 | **34.96** |
| `isotropy_score` | 0.036 | **0.100** |
| Severidad | HIGH | HIGH |
| `suggested_target_dim` | 16 | 28 |

Los embeddings preentrenados son **menos anisotrópicos y más ricos**: su
energía se reparte entre ~35 direcciones efectivas (vs. ~8 en píxeles), y el
diagnóstico sugiere reducir a 28 dimensiones (vs. 16). Es decir, la red
preentrenada distribuye la información en más ejes útiles que los píxeles
crudos, donde casi todo se concentra en unas pocas direcciones de brillo/forma.
Este cambio de geometría explica varios de los resultados de abajo (en
particular, por qué PCA a pocas componentes pierde correlación de distancias
sobre embeddings).

## 4. Resultados — Sección A: proyecciones 2D/3D y métricas

Métricas de calidad de cada proyección (definiciones idénticas al experimento
crudo: Trustworthiness y Continuity son no supervisadas; kNN acc y Silhouette
usan `y` sólo para *evaluar*; Corr. distancias es la preservación de la
estructura global). **Entre paréntesis, el valor sobre píxeles crudos.**

| Método (entrada = embeddings) | Trustworthiness | Continuity | kNN acc | Silhouette | Corr. distancias |
|---|---|---|---|---|---|
| PCA | 0.811 (0.917) | 0.930 (0.972) | 0.500 (0.525) | 0.032 (−0.030) | 0.523 (**0.880**) |
| **UMAP** | 0.959 (0.980) | 0.974 (0.983) | **0.780** (0.741) | **0.281** (0.177) | 0.473 (0.625) |
| EmbedKit self-sup (2D) | 0.829 (0.805) | 0.950 (0.953) | 0.639 (0.491) | 0.163 (−0.027) | 0.434 (0.444) |
| EmbedKit self-sup (3D) | 0.926 (0.954) | 0.974 (0.986) | 0.713 (0.651) | 0.139 (0.056) | 0.439 (0.588) |
| EmbedKit superv. (3D) *(ref., usa `y`)* | 0.842 (0.840) | 0.937 (0.940) | **0.965** (0.931) | **0.659** (0.601) | 0.427 (0.512) |

(CSV completo: `resultados_fashion_mnist_embeddings/tabla_metricas_2d.csv`;
figura: `resultados_fashion_mnist_embeddings/fig_embeddings_2d.png`)

**Lo que mejora con embeddings (separación de clases):**
- **Silhouette sube en todos los métodos.** Con píxeles, PCA y EmbedKit
  self-sup (2D) tenían Silhouette **negativo** (−0.030, −0.027); con embeddings
  pasan a positivo (0.032, 0.163). UMAP sube de 0.177 a **0.281**. Las clases
  están mejor separadas en el espacio de embeddings.
- **UMAP gana en kNN acc** (0.780 vs. 0.741) y produce el mejor mapa 2D no
  supervisado: en `fig_embeddings_2d.png` separa islas limpias para *Bag*,
  *Trouser* y el bloque de calzado (*Sandal/Sneaker/Ankle boot*), dejando
  juntas sólo las clases visualmente ambiguas (*T-shirt/Shirt/Coat/Pullover/
  Dress*), que es exactamente donde Fashion-MNIST es difícil.
- **EmbedKit self-sup (2D) deja de ser basura en 2D:** su kNN acc sube de 0.491
  a 0.639 y su Silhouette de negativo a positivo. Aun así, **sigue colapsando
  a un anillo S¹** (ver la figura) — la patología del experimento original no
  desaparece, sólo se atenúa.

**Lo que empeora con embeddings (estructura global):**
- **PCA pierde correlación de distancias:** de **0.880** (píxeles) a **0.523**
  (embeddings) en 2D. Es consecuencia directa de la Sección 3: como la energía
  de los embeddings está repartida en ~35 direcciones (no en ~8), dos
  componentes principales capturan una fracción mucho menor de la geometría
  global. PCA seguía siendo el rey de la estructura global en píxeles; sobre
  embeddings ya no.

### Sección A a 16 dimensiones

Igual que en el experimento original, se repite dándole a PCA/UMAP/EmbedKit
la dimensión sugerida por el diagnóstico (aquí se usó **16**, para comparar
directamente con la tabla de 16D del experimento crudo):

| Método (16D, entrada = embeddings) | Trustworthiness | Continuity | kNN acc | Silhouette | Corr. distancias |
|---|---|---|---|---|---|
| PCA (16D) | 0.991 (0.995) | 0.996 (0.998) | 0.801 (0.768) | 0.145 (0.076) | 0.795 (**0.981**) |
| UMAP (16D) | 0.978 (0.985) | 0.978 (0.986) | 0.791 (0.737) | 0.279 (0.178) | 0.484 (0.612) |
| EmbedKit self-sup (16D) | 0.991 (0.994) | 0.995 (0.997) | **0.808** (0.782) | 0.080 (0.048) | 0.403 (0.297) |
| EmbedKit superv. (16D) *(ref., usa `y`)* | 0.943 (0.952) | 0.963 (0.968) | **0.974** (0.941) | 0.500 (0.449) | 0.448 (0.544) |

A 16D, los tres métodos no supervisados suben su kNN acc respecto a píxeles
(PCA 0.801 vs 0.768, UMAP 0.791 vs 0.737, EmbedKit self-sup 0.808 vs 0.782):
**con suficiente dimensión, la mejor separación del embedding se traduce en
mejor exactitud**. La excepción vuelve a ser la correlación de distancias de
PCA (0.795 vs 0.981): la estructura global sigue siendo más difícil de
comprimir linealmente en el espacio de embeddings.

## 5. Resultados — Sección B: generalización fuera de muestra (el resultado central)

Prueba clave (5-fold estratificado, `DIM_GEN=3`): cada encoder se ajusta
**sólo** con el train de cada fold, se proyecta el test, y se mide un k-NN
entrenado y evaluado en el espacio proyectado. **Entre paréntesis, píxeles
crudos.**

| Método | OOS kNN acc (media ± std) | OOS Trustworthiness |
|---|---|---|
| PCA | 0.637 ± 0.013 (0.620 ± 0.018) | 0.881 (0.951) |
| **UMAP** | **0.785 ± 0.009** (0.726 ± 0.009) | 0.955 (0.975) |
| **EmbedKit self-sup** | **0.235 ± 0.242** (0.345 ± 0.235) | 0.587 (0.703) |
| EmbedKit superv. *(ref., usa `y`)* | 0.628 ± 0.095 (0.680 ± 0.109) | 0.792 (0.825) |
| **Base (sin reducir)** | **0.813 ± 0.008** (R¹²⁸⁰)  vs  0.774 ± 0.016 (R⁷⁸⁴) | — |

(CSV: `resultados_fashion_mnist_embeddings/generalizacion_oos.csv`; figura:
`resultados_fashion_mnist_embeddings/fig_generalizacion.png`)

**Tres conclusiones:**

1. **El embedding preentrenado es una mejor representación base.** El k-NN
   directo sobre los embeddings sin reducir generaliza mejor que sobre los
   píxeles crudos: **0.813 vs. 0.774** (misma muestra, misma validación). La
   red de ImageNet aporta información útil a Fashion-MNIST pese a no haber sido
   entrenada en ropa.

2. **UMAP es quien más aprovecha el embedding y sigue siendo estable.** Su
   exactitud fuera de muestra sube de **0.726 → 0.785** (+5.9 puntos) y su
   desviación entre folds sigue siendo mínima (0.009). Es el mejor método **no
   supervisado** de reducción, y queda a un paso del k-NN base sin reducir,
   pero comprimiendo a sólo 3 dimensiones.

3. **El hallazgo central del experimento original se mantiene: EmbedKit
   self-sup sigue siendo inestable fuera de muestra.** Con embeddings su media
   es **0.235 ± 0.242** — colapsa en varios folds igual que con píxeles (allá
   era 0.345 ± 0.235), con una desviación estándar **~27 veces mayor que la de
   UMAP**. Es decir: **cambiar la entrada de píxeles a embeddings ricos NO cura
   la inestabilidad del refinador contrastivo no supervisado al reentrenarse en
   cada fold.** Mejores insumos suben el techo (mejor Silhouette, mejor 2D
   in-sample), pero el problema de generalización del método persiste.

## 6. Barrido de hiperparámetros de EmbedKit sobre los embeddings

El experimento original incluía un **barrido por descenso de coordenadas**
(`embedkit_unsup_sweep_fashion_mnist.py`) que buscaba, sobre píxeles crudos,
una configuración no supervisada de EmbedKit que superara a UMAP — y la
encontraba (`FeatureMasking`, red más grande → **0.775 ± 0.016**, batía a UMAP
y **curaba la inestabilidad**). Aquí se repite **el mismo barrido, con las
mismas etapas**, pero alimentándolo con los embeddings preentrenados
(`embedkit_unsup_sweep_embeddings.py`). Cada configuración se rankea por
exactitud k-NN fuera de muestra en un holdout 75/25; los finalistas se
re-evalúan con **5-fold × 3 semillas**.

**Etapas del barrido** (holdout; referencia UMAP ≈ 0.787 a `dim=16`):

| Etapa | Se explora | Ganador (embeddings) | OOS acc (holdout) | Ganador en píxeles |
|---|---|---|---|---|
| 1. `target_dim` | 8, 16, 32 | **16** | 0.796 | 32 |
| 2. augmentation | KNNPairs(5/10/15/hn), Noise, Masking, Mixup, KNN+Noise | **KNNPairs(k=10)** | 0.807 | FeatureMasking |
| 3. loss / temp. | NTXent(.05/.07/.2/.5), AlignUniform, Combined×2 | **NTXent(0.07)** | 0.800 | Combined |
| 4. arquitectura | hidden, n_layers, lr, epochs | **base** (sin cambios) | 0.819 | h512/L3/lr1e-3 |

(CSVs por etapa en `resultados_sweep_embeddings/etapa*.csv`)

**La observación más importante del barrido: el mecanismo decisivo se invierte
respecto a los píxeles.** En píxeles, `FeatureMasking` (enmascarar parches)
ganaba y `KNNPairs` era la *augmentation* problemática; sobre embeddings es al
revés — **`KNNPairs` es la mejor** (0.79–0.81) y **`FeatureMasking` cae a
0.723**, entre las peores. Es coherente: en el espacio de embeddings los
vecindarios son semánticamente significativos, así que fabricar positivos con
vecinos (`KNNPairs`) es fiable; en cambio enmascarar *features aprendidas*
destruye información, al contrario que enmascarar píxeles redundantes.
`GaussianNoise` vuelve a ser de las peores (0.547) en ambas representaciones.

**Finalistas (5-fold × 3 semillas, `target_dim=16`) — y el resultado clave:**

| Config | OOS kNN acc (media ± std) | OOS Trustworthiness |
|---|---|---|
| **EmbedKit default** (KNNPairs(5), Combined, h256/L2) | **0.803 ± 0.015** | 0.974 |
| EmbedKit **sweep-best** (KNNPairs(10), NTXent.07) | 0.619 ± 0.264 | 0.844 |
| UMAP (dim=16) | 0.786 ± 0.016 | 0.958 |

Comparado con el barrido sobre píxeles crudos:

| Config | Píxeles (dim=32) | Embeddings (dim=16) |
|---|---|---|
| EmbedKit default | 0.611 ± 0.062 | **0.803 ± 0.015** |
| EmbedKit sweep-best | **0.775 ± 0.016** | 0.619 ± 0.264 |
| UMAP | 0.729 ± 0.020 | 0.786 ± 0.016 |

(CSV: `resultados_sweep_embeddings/finalistas.csv`; figura:
`resultados_sweep_embeddings/fig_finalistas.png`; config completa en
`resultados_sweep_embeddings/resumen.json`)

**Dos conclusiones, y son distintas a las del barrido sobre píxeles:**

1. **Sobre embeddings ya no hace falta afinar: la configuración por defecto de
   EmbedKit supera a UMAP y es estable** (0.803 ± 0.015 vs. 0.786 ± 0.016).
   Sobre píxeles, la configuración por defecto era inestable (0.611 ± 0.062) y
   había que *buscar* `FeatureMasking` para batir a UMAP; sobre embeddings, la
   representación más rica ya hace ese trabajo — con la dimensión adecuada (16),
   la EmbedKit **por defecto** es estable y ganadora. (Nótese que a `dim=3`, en
   la Sección 5, esa misma EmbedKit self-sup colapsaba: la clave es
   representación rica **+** dimensión suficiente.)

2. **El barrido guiado por un solo holdout es contraproducente aquí.** La
   "config ganadora" que eligió el descenso de coordenadas (KNNPairs(10) +
   NTXent(.07), mejor por centésimas en el holdout, 0.807) **colapsa en la
   evaluación robusta**: 0.619 ± 0.264 — peor y ~18× más inestable que la
   propia config por defecto. Un cambio mínimo (k=5→10 y cambiar la loss) volvió
   inestable una config estable. Esto revela dos cosas: (a) la selección por un
   único holdout es ruidosa —agravada por la no-determinación de PyTorch en CPU,
   que hace que hasta re-evaluar la *misma* config varíe entre 0.80 y 0.82—, y
   (b) la estabilidad del refinador self-supervised sigue siendo **frágil** ante
   la configuración, aunque su valor por defecto sea bueno. En resumen: sobre
   esta representación, lo mejor es **no** correr el barrido.

## 7. Conclusión

Usar embeddings de una red preentrenada como entrada, en vez de píxeles crudos:

- **Sube el techo de todos los métodos** en separación de clases (Silhouette),
  en kNN acc (sobre todo a 16D) y en la representación base (0.774 → 0.813).
  UMAP es el mayor beneficiado en 2D y produce el mejor mapa no supervisado.
- **Cambia la geometría**: los embeddings son menos anisotrópicos, lo que
  quita a PCA su ventaja en estructura global (correlación de distancias
  0.880 → 0.523 en 2D).
- **Estabiliza a EmbedKit a su dimensión cómoda (16):** lo que en píxeles exigía
  afinar la *augmentation* (Sección 6), en embeddings ya lo logra la
  configuración por defecto — que pasa a **superar a UMAP de forma estable**
  (0.803 vs. 0.786). El eje que en el experimento anterior parecía irresoluble
  (la inestabilidad de EmbedKit self-sup) se resuelve **mejorando la
  representación de entrada**, no los hiperparámetros.
- **Matiz:** a dimensión baja (2D–3D) EmbedKit self-sup sigue colapsando fuera
  de muestra (Sección 5), y el barrido por holdout único puede empeorar las
  cosas. La mejora viene del binomio *representación rica + dimensión
  suficiente*, no de cada pieza por separado.

## 8. Cómo reproducir

```bash
cd experiments
pip install -r requirements.txt        # incluye tensorflow para el extractor

# Corrida por defecto (N=3000, SEED=0, K=10, epochs=150, MobileNetV2, 224x224):
python fashion_mnist_embeddings_preentrenadas.py

# Otras redes preentrenadas:
python fashion_mnist_embeddings_preentrenadas.py --modelo resnet50
python fashion_mnist_embeddings_preentrenadas.py --modelo efficientnet_b0

# Sólo extraer y cachear los embeddings (paso 1, opcional):
python extraer_embeddings_preentrenados.py --n 3000 --modelo mobilenet_v2 \
    --img-size 224 --out .cache_embeddings/emb_mobilenet_v2_n3000_s0_img224.npz

# Barrido de HP de EmbedKit sobre los embeddings (Sección 6):
python embedkit_unsup_sweep_embeddings.py            # completo (~45 min)
python embedkit_unsup_sweep_embeddings.py --quick    # reducido para pruebas
```

Salidas del experimento principal en `resultados_fashion_mnist_embeddings/`:
`tabla_metricas_2d.csv`, `generalizacion_oos.csv`, `config.json` y las figuras
`fig_embeddings_2d.png`, `fig_metricas_barras.png`, `fig_generalizacion.png`.
Salidas del barrido en `resultados_sweep_embeddings/`: `etapa*.csv`,
`finalistas.csv`, `resumen.json`, `fig_finalistas.png`.
Los embeddings se cachean en `.cache_embeddings/` (ignorado por git) para no
recalcularlos entre corridas.

> **Configuración de la corrida reportada:** `N=3000`, `SEED=0`, `K=10`,
> `epochs=150`, `DIM_GEN=3`, `dim-alt=16`, `modelo=mobilenet_v2`,
> `img-size=224`. Ver `resultados_fashion_mnist_embeddings/config.json`.
