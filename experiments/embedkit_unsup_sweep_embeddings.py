"""
Barrido de configuraciones de refinamiento NO SUPERVISADO de EmbedKit sobre los
EMBEDDINGS preentrenados de Fashion-MNIST (replica del sweep de pixeles crudos).

Es exactamente el mismo barrido que `embedkit_unsup_sweep_fashion_mnist.py`
(mismas etapas de descenso por coordenadas: target_dim -> augmentation ->
loss/temperatura -> arquitectura, y los mismos finalistas 5-fold x semillas vs
UMAP), cambiando UNA sola cosa: la entrada del refinador y de UMAP ya no son los
pixeles crudos (R^784) sino los embeddings de una red preentrenada en ImageNet
(por defecto MobileNetV2 -> R^1280), calculados con `include_top=False` +
global average pooling.

Pregunta: sobre esta representacion mas rica, ¿logra alguna configuracion del
refinador self-supervised de EmbedKit **superar a UMAP** en generalizacion
fuera de muestra? ¿Y sigue siendo tan inestable como sobre los pixeles crudos?

Arquitectura en dos pasos (igual que el experimento de embeddings): la
extraccion de embeddings (TensorFlow) corre en un subproceso aparte y se cachea
en `.npz`; este script (PyTorch/UMAP/sklearn) lee ese cache y corre el barrido.

Uso:
    python embedkit_unsup_sweep_embeddings.py [--n 3000] [--seed 0] [--k 10]
        [--epochs 150] [--modelo mobilenet_v2] [--img-size 224]
        [--outdir resultados_sweep_embeddings] [--quick]
"""

import argparse
import sys
import warnings
from pathlib import Path

# Ver nota en fashion_mnist_embeddings_preentrenadas.py: bloqueamos TensorFlow
# ANTES de importar cualquier cosa que arrastre umap, para que no choque con
# PyTorch. El extractor de embeddings (que si usa TF) corre en un subproceso.
sys.modules.setdefault("tensorflow", None)

import numpy as np

# Reutiliza el barrido completo (etapas 1-5) y la obtencion de embeddings.
from embedkit_unsup_sweep_fashion_mnist import correr_sweep
from fashion_mnist_embeddings_preentrenadas import obtener_embeddings

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument(
        "--modelo", type=str, default="mobilenet_v2",
        choices=["mobilenet_v2", "resnet50", "efficientnet_b0"],
        help="red preentrenada (ImageNet) para extraer embeddings",
    )
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--data-dir", type=str, default=".data")
    ap.add_argument("--cache-dir", type=str, default=".cache_embeddings")
    ap.add_argument("--outdir", type=str, default="resultados_sweep_embeddings")
    ap.add_argument("--quick", action="store_true", help="barrido reducido para pruebas")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 74)
    print("Barrido de refinamiento NO SUPERVISADO de EmbedKit vs UMAP")
    print(f"sobre EMBEDDINGS preentrenados ({args.modelo}, ImageNet) de Fashion-MNIST")
    print("=" * 74)

    # La UNICA diferencia con el sweep original: X son embeddings, no pixeles.
    X, y, dim = obtener_embeddings(
        args.n, args.seed, args.modelo, args.img_size, args.data_dir, args.cache_dir
    )
    print(f"Entrada del barrido: embeddings R^{dim} ({X.shape[0]} puntos)")

    # Mismo barrido, misma logica; solo cambia la representacion de entrada.
    correr_sweep(X, y, args, outdir)


if __name__ == "__main__":
    main()
