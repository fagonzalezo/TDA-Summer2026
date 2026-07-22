"""
Extractor de embeddings preentrenados para Fashion-MNIST (paso 1 del
experimento de embeddings).

Este script se ejecuta como PROCESO APARTE y solo importa TensorFlow: calcula
los embeddings de una submuestra estratificada de Fashion-MNIST con una red
preentrenada en ImageNet (`tf.keras.applications`) y los guarda en un `.npz`.

Se aisla en su propio proceso a proposito: TensorFlow y PyTorch (que usa
embedding-kit en el paso 2) comparten librerias OpenMP/MKL y, cargados en el
mismo interprete, provocan un `Segmentation fault`. Separarlos evita el
choque: aqui solo vive TensorFlow; el paso 2 (`fashion_mnist_embeddings_
preentrenadas.py`) solo usa PyTorch y lee el `.npz` que produce este script.

La red se usa con `include_top=False, pooling='avg'`, de modo que la salida es
el vector tras el global average pooling: la representacion que la red aprendio
en ImageNet, sin la capa de clasificacion. Ese vector es el embedding.

Uso (normalmente invocado por el paso 2, pero corre solo):
    python extraer_embeddings_preentrenados.py --n 3000 --seed 0 \
        --modelo mobilenet_v2 --img-size 224 --out cache_embeddings.npz
"""

import argparse
import gzip
import struct
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Datos (sin PyTorch): keras primero; IDX en disco como respaldo
# ---------------------------------------------------------------------------

def _leer_idx_imagenes(ruta):
    with gzip.open(ruta, "rb") as f:
        b = f.read()
    _, n, r, c = struct.unpack(">IIII", b[:16])
    return np.frombuffer(b[16:], dtype=np.uint8).reshape(n, r * c)


def _leer_idx_etiquetas(ruta):
    with gzip.open(ruta, "rb") as f:
        b = f.read()
    _, n = struct.unpack(">II", b[:8])
    return np.frombuffer(b[8:], dtype=np.uint8)


def cargar_fashion_mnist(data_dir=".data"):
    """Fashion-MNIST (split de entrenamiento, 60k) en el MISMO orden que usa el
    experimento crudo (torchvision/IDX), con pixeles escalados a [0, 1].

    Primero intenta el loader de keras (descarga desde storage.googleapis.com);
    si no hay red, lee los IDX que torchvision ya dejo cacheados en `.data`.
    Ambas fuentes comparten el orden canonico del split de entrenamiento, asi
    que la submuestra estratificada coincide con la del experimento de pixeles.
    """
    try:
        from tensorflow.keras.datasets import fashion_mnist

        (x_tr, y_tr), _ = fashion_mnist.load_data()
        X_full = x_tr.reshape(len(x_tr), -1).astype("float32") / 255.0
        return X_full, y_tr.astype(int)
    except Exception as e:  # pragma: no cover - respaldo offline
        print(f"keras.datasets fallo ({e}); leyendo IDX cacheados de {data_dir}...")
        raw = Path(data_dir) / "FashionMNIST" / "raw"
        X_full = _leer_idx_imagenes(raw / "train-images-idx3-ubyte.gz").astype("float32") / 255.0
        y_full = _leer_idx_etiquetas(raw / "train-labels-idx1-ubyte.gz").astype(int)
        return X_full, y_full


def submuestra_estratificada(X_full, y_full, n, seed):
    """Identica a la del experimento crudo: N puntos, misma cantidad por clase.

    Devuelve tambien los indices seleccionados para poder reconstruir la MISMA
    muestra en el experimento de pixeles crudos.
    """
    rng = np.random.RandomState(seed)
    n_clases = len(np.unique(y_full))
    por_clase = n // n_clases
    sel = np.hstack(
        [rng.choice(np.where(y_full == c)[0], por_clase, replace=False) for c in range(n_clases)]
    )
    rng.shuffle(sel)
    return X_full[sel], y_full[sel], sel


# ---------------------------------------------------------------------------
# 2. Extractor preentrenado (tf.keras.applications)
# ---------------------------------------------------------------------------

def construir_extractor(modelo, img_size):
    """Devuelve (red_extractora, preprocesamiento, dim_embedding)."""
    from tensorflow.keras import applications as kapp

    mapa = {
        "mobilenet_v2": (kapp.MobileNetV2, kapp.mobilenet_v2.preprocess_input),
        "resnet50": (kapp.ResNet50, kapp.resnet.preprocess_input),
        "efficientnet_b0": (kapp.EfficientNetB0, kapp.efficientnet.preprocess_input),
    }
    if modelo not in mapa:
        raise ValueError(f"Modelo no soportado: {modelo}. Opciones: {list(mapa)}")
    constructor, preprocesar = mapa[modelo]
    base = constructor(
        include_top=False, weights="imagenet",
        input_shape=(img_size, img_size, 3), pooling="avg",
    )
    base.trainable = False
    return base, preprocesar, int(base.output_shape[-1])


def calcular_embeddings(X_raw, modelo, img_size, batch, seed):
    """X_raw (N,784) en [0,1] -> (N,d) embeddings preentrenados (post GAP)."""
    import tensorflow as tf

    tf.random.set_seed(seed)
    tf.config.set_visible_devices([], "GPU")  # CPU: reproducible y sin drivers

    base, preprocesar, dim = construir_extractor(modelo, img_size)
    N = X_raw.shape[0]
    imgs = X_raw.reshape(N, 28, 28).astype("float32")

    print(f"Extrayendo embeddings con {modelo} (ImageNet) -> R^{dim}; "
          f"{N} imagenes a {img_size}x{img_size}...")
    salidas = []
    t0 = time.time()
    for i in range(0, N, batch):
        lote = imgs[i : i + batch]                    # (b,28,28) en [0,1]
        lote = np.repeat(lote[..., None], 3, axis=-1)  # gris -> RGB
        lote = tf.image.resize(lote, (img_size, img_size))
        lote = preprocesar(lote * 255.0)               # a [0,255] + preproc del modelo
        salidas.append(base(lote, training=False).numpy())
    X_emb = np.vstack(salidas).astype("float32")
    print(f"Embeddings listos: {X_emb.shape} en {time.time() - t0:.1f} s")
    return X_emb, dim


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modelo", type=str, default="mobilenet_v2",
                    choices=["mobilenet_v2", "resnet50", "efficientnet_b0"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--data-dir", type=str, default=".data")
    ap.add_argument("--out", type=str, required=True, help="ruta del .npz de salida")
    args = ap.parse_args()

    X_full, y_full = cargar_fashion_mnist(args.data_dir)
    print(f"Fashion-MNIST completo: {X_full.shape}, clases: {sorted(np.unique(y_full))}")
    X_raw, y, sel = submuestra_estratificada(X_full, y_full, args.n, args.seed)
    print(f"Submuestra: {X_raw.shape} ({args.n // len(np.unique(y_full))} por clase)")

    X_emb, dim = calcular_embeddings(X_raw, args.modelo, args.img_size, args.batch, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X_emb=X_emb, y=y, X_raw=X_raw.astype("float32"), sel=sel,
        modelo=args.modelo, img_size=args.img_size, seed=args.seed, dim=dim,
    )
    print(f"Guardado: {args.out}  (X_emb={X_emb.shape}, X_raw={X_raw.shape})")


if __name__ == "__main__":
    main()
