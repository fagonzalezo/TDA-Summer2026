"""
Experimento: reduccion de dimensionalidad sobre Fashion-MNIST usando como
entrada EMBEDDINGS de un modelo de Deep Learning preentrenado (ImageNet).

Replica exactamente la metodologia de `fashion_mnist_umap_vs_embedkit.py`
(mismo N, SEED, K, muestreo estratificado, metricas, secciones A y B y
graficas), cambiando UNA sola cosa: en lugar de alimentar a los modelos con
los pixeles crudos (R^784), primero se calculan embeddings de cada imagen con
una red convolucional preentrenada (por defecto MobileNetV2 sobre ImageNet) y
esos embeddings son la entrada de PCA / UMAP / EmbedKit y del k-NN de
referencia.

Arquitectura en dos pasos (a proposito):
  1. `extraer_embeddings_preentrenados.py` corre como PROCESO APARTE (solo
     TensorFlow) y cachea los embeddings en un `.npz`. Se aisla porque
     TensorFlow y PyTorch en el mismo interprete chocan (Segmentation fault).
  2. Este script (solo PyTorch/embedding-kit/UMAP/sklearn) lee ese `.npz` y
     corre las secciones A y B identicas al experimento de pixeles crudos.

Pregunta: ¿mejora la calidad de las proyecciones y la generalizacion cuando
los modelos parten de representaciones semanticas de una red preentrenada en
lugar de pixeles crudos?

Uso:
    python fashion_mnist_embeddings_preentrenadas.py \
        [--n 3000] [--seed 0] [--k 10] [--epochs 150] [--dim-gen 3] \
        [--modelo mobilenet_v2] [--img-size 224] [--dim-alt 16] \
        [--outdir resultados_fashion_mnist_embeddings] [--con-tsne] [--sin-supervisado]

Requiere `tensorflow` (extractor, paso 1) ademas de las dependencias del
experimento original (paso 2; ver requirements.txt).
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

# IMPORTANTE: en este proceso conviven UMAP y PyTorch (embedding-kit). Algunas
# versiones de umap-learn importan TensorFlow de forma ansiosa (para Parametric
# UMAP) si esta instalado, y TF + PyTorch en el mismo interprete provocan un
# Segmentation fault en este entorno. Bloqueamos el modulo `tensorflow` ANTES de
# importar umap: umap detecta el ImportError y sigue con su UMAP normal. La
# extraccion de embeddings (que si necesita TF) corre en un subproceso aparte,
# con su propio interprete, asi que este bloqueo no la afecta.
sys.modules.setdefault("tensorflow", None)

import numpy as np

# Se reutilizan tal cual las piezas del experimento original para que la
# comparacion pixeles-crudos vs embeddings-preentrenados sea directa.
# (Este import trae PyTorch/embedding-kit y, transitivamente, umap.)
from fashion_mnist_umap_vs_embedkit import (
    HAY_EMBEDKIT,
    HAY_UMAP,
    correr_seccion_a,
    correr_seccion_b_generalizacion,
    graficar_embeddings_2d,
    graficar_generalizacion,
    graficar_metricas_barras,
)

warnings.filterwarnings("ignore")

_AQUI = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Embeddings preentrenados (via subproceso TensorFlow aislado, con cache)
# ---------------------------------------------------------------------------

def obtener_embeddings(n, seed, modelo, img_size, data_dir, cache_dir):
    """Devuelve (X_emb, y, dim) usando un cache `.npz`.

    Si el cache no existe, lanza `extraer_embeddings_preentrenados.py` como
    proceso aparte (solo TensorFlow) para evitar el choque TF+PyTorch.
    """
    cache = Path(cache_dir) / f"emb_{modelo}_n{n}_s{seed}_img{img_size}.npz"
    if not cache.exists():
        print(f"No hay cache; extrayendo embeddings en subproceso -> {cache}")
        cmd = [
            sys.executable, str(_AQUI / "extraer_embeddings_preentrenados.py"),
            "--n", str(n), "--seed", str(seed), "--modelo", modelo,
            "--img-size", str(img_size), "--data-dir", str(data_dir),
            "--out", str(cache),
        ]
        subprocess.run(cmd, check=True, cwd=str(_AQUI))
    else:
        print(f"Usando embeddings cacheados: {cache}")

    datos = np.load(cache, allow_pickle=True)
    X_emb = datos["X_emb"].astype("float32")
    y = datos["y"].astype(int)
    dim = int(datos["dim"])
    print(f"Embeddings: {X_emb.shape} (R^{dim}); etiquetas: {y.shape}")
    return X_emb, y, dim


# ---------------------------------------------------------------------------
# 2. Main (mismo flujo que el experimento original, con X = embeddings)
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000, help="tamano de la submuestra estratificada")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10, help="numero de vecinos comun")
    ap.add_argument("--epochs", type=int, default=150, help="epocas de entrenamiento de EmbedKit")
    ap.add_argument("--dim-gen", type=int, default=3, help="dimension para la prueba de generalizacion")
    ap.add_argument(
        "--dim-alt", type=int, default=16,
        help="dimension adicional para PCA/UMAP/EmbedKit en la Seccion A; 0 para omitirla",
    )
    ap.add_argument(
        "--modelo", type=str, default="mobilenet_v2",
        choices=["mobilenet_v2", "resnet50", "efficientnet_b0"],
        help="red preentrenada (ImageNet, tf.keras.applications) para extraer embeddings",
    )
    ap.add_argument("--img-size", type=int, default=224, help="tamano de entrada de la red preentrenada")
    ap.add_argument("--data-dir", type=str, default=".data", help="cache de Fashion-MNIST (IDX)")
    ap.add_argument("--cache-dir", type=str, default=".cache_embeddings", help="cache de los .npz de embeddings")
    ap.add_argument("--outdir", type=str, default="resultados_fashion_mnist_embeddings")
    ap.add_argument("--con-tsne", action="store_true", help="incluir t-SNE (mas lento)")
    ap.add_argument(
        "--sin-supervisado", action="store_true",
        help="omitir EmbedKit supervisado (referencia con etiquetas)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    con_supervisado = not args.sin_supervisado

    print("=" * 74)
    print("Experimento: reduccion de dimensionalidad sobre EMBEDDINGS")
    print(f"preentrenados ({args.modelo}, ImageNet) de Fashion-MNIST")
    print("=" * 74)

    # (a) Embeddings de la MISMA submuestra estratificada que el experimento crudo.
    #     La UNICA diferencia con el original: la entrada de los modelos son
    #     embeddings de una red preentrenada, no pixeles crudos (R^784).
    X, y, dim_emb = obtener_embeddings(
        args.n, args.seed, args.modelo, args.img_size, args.data_dir, args.cache_dir
    )

    if HAY_EMBEDKIT:
        from embedkit import EmbedKitAnalyzer

        print("\n--- Diagnostico EmbedKitAnalyzer sobre los embeddings preentrenados ---")
        EmbedKitAnalyzer(k=args.k).fit(X.astype("float32")).print_summary()

    print("\n--- Seccion A: proyecciones 2D/3D de los embeddings y metricas ---")
    tabla_a, embeddings = correr_seccion_a(
        X, y, args.k, args.epochs, args.con_tsne, con_supervisado, args.seed,
        dim_alt=args.dim_alt or None,
    )
    print(tabla_a)
    tabla_a.to_csv(outdir / "tabla_metricas_2d.csv")

    print("\n--- Seccion B: generalizacion fuera de muestra ---")
    tabla_b = correr_seccion_b_generalizacion(
        X, y, args.k, args.dim_gen, args.epochs, con_supervisado, args.seed
    )
    # La fila base es el embedding preentrenado sin reducir (no R^784 de pixeles).
    tabla_b = tabla_b.rename(
        index={"Base (R^784, sin embeber)": f"Base (R^{dim_emb}, embeddings preentrenados sin reducir)"}
    )
    print(tabla_b)
    tabla_b.to_csv(outdir / "generalizacion_oos.csv")

    print("\n--- Graficas ---")
    graficar_embeddings_2d(
        {n: e for n, e in embeddings.items() if e.shape[1] >= 2}, y, outdir
    )
    graficar_metricas_barras(tabla_a, outdir)
    graficar_generalizacion(tabla_b, outdir)

    config = vars(args) | {
        "hay_umap": HAY_UMAP,
        "hay_embedkit": HAY_EMBEDKIT,
        "dim_embedding": int(dim_emb),
        "entrada_modelos": f"embeddings preentrenados {args.modelo} (ImageNet)",
    }
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nListo. Resultados en: {outdir.resolve()}")


if __name__ == "__main__":
    main()
