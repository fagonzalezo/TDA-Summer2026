"""
Experimento: UMAP vs EmbedKit (no supervisado) en Fashion-MNIST.

Replica la metodologia del notebook `UNAL_Bog_TDA_Comparacion_Reduccion_Dimensionalidad.ipynb`
(mismo K, SEED, muestreo estratificado, metricas y prueba de generalizacion fuera de
muestra) mas alla de `load_digits()`, sobre un dataset mas complejo y de mayor
dimension: Fashion-MNIST (784-dim, 10 clases, clases visualmente solapadas).

Pregunta: ¿UMAP sigue comportandose mejor que el refinador no supervisado
(`self_supervised`) de embedding-kit cuando el dataset es mas dificil?

Uso:
    python fashion_mnist_umap_vs_embedkit.py [--n 3000] [--seed 0] [--k 10]
        [--epochs 150] [--dim-gen 3] [--outdir resultados_fashion_mnist]
        [--con-tsne] [--sin-supervisado]
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

try:
    import umap

    HAY_UMAP = True
except ImportError:
    HAY_UMAP = False

try:
    from embedkit import EmbedKit, EmbedKitAnalyzer

    HAY_EMBEDKIT = True
except ImportError:
    HAY_EMBEDKIT = False


# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------

def cargar_fashion_mnist():
    """Descarga Fashion-MNIST (784-dim, 10 clases) y escala pixeles a [0, 1]."""
    try:
        from sklearn.datasets import fetch_openml

        print("Descargando Fashion-MNIST via OpenML (puede tardar la primera vez)...")
        datos = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
        X_full = datos.data.astype("float32") / 255.0
        y_full = datos.target.astype(int)
        return X_full, y_full
    except Exception as e:
        print(f"fetch_openml fallo ({e}); probando con torchvision...")
        from torchvision.datasets import FashionMNIST

        ds = FashionMNIST(root=".data", train=True, download=True)
        X_full = ds.data.numpy().reshape(len(ds), -1).astype("float32") / 255.0
        y_full = ds.targets.numpy().astype(int)
        return X_full, y_full


def submuestra_estratificada(X_full, y_full, n, seed):
    """Toma N puntos, misma cantidad por clase, igual que el notebook de digits."""
    rng = np.random.RandomState(seed)
    n_clases = len(np.unique(y_full))
    por_clase = n // n_clases
    sel = np.hstack(
        [rng.choice(np.where(y_full == c)[0], por_clase, replace=False) for c in range(n_clases)]
    )
    rng.shuffle(sel)
    print(f"Puntos por clase: {por_clase} (total {len(sel)})")
    return X_full[sel], y_full[sel]


NOMBRES_CLASES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# ---------------------------------------------------------------------------
# 2. Metricas (identicas al notebook de digits)
# ---------------------------------------------------------------------------

def evaluar(X_alto, X_bajo, y, k):
    """Calcula las metricas de calidad de una proyeccion X_bajo de X_alto.

    1-2 y 5 son no supervisadas (no usan y); 3-4 usan las etiquetas.
    """
    tw = trustworthiness(X_alto, X_bajo, n_neighbors=k)
    cont = trustworthiness(X_bajo, X_alto, n_neighbors=k)

    knn = KNeighborsClassifier(n_neighbors=k)
    acc = cross_val_score(knn, X_bajo, y, cv=5).mean()

    sil = silhouette_score(X_bajo, y)

    rho = spearmanr(pdist(X_alto), pdist(X_bajo)).correlation

    return {
        "Trustworthiness": tw,
        "Continuity": cont,
        "kNN acc": acc,
        "Silhouette": sil,
        "Corr. distancias": rho,
    }


# ---------------------------------------------------------------------------
# 3. Seccion A: embeddings y tabla comparativa
# ---------------------------------------------------------------------------

def correr_seccion_a(X, y, k, epochs, con_tsne, con_supervisado, seed):
    filas = {}
    embeddings = {}

    tecnicas_2d = {"PCA": PCA(n_components=2, random_state=seed)}
    if HAY_UMAP:
        tecnicas_2d["UMAP"] = umap.UMAP(n_components=2, n_neighbors=k, random_state=seed)
    else:
        print("UMAP no esta instalado; se omitira.")

    if con_tsne:
        from sklearn.manifold import TSNE

        tecnicas_2d["t-SNE"] = TSNE(n_components=2, random_state=seed, init="pca", perplexity=k)

    for nombre, modelo in tecnicas_2d.items():
        t0 = time.time()
        X_2d = modelo.fit_transform(X)
        dt = time.time() - t0
        metricas = evaluar(X, X_2d, y, k)
        metricas["Tiempo (s)"] = dt
        filas[nombre] = metricas
        embeddings[nombre] = X_2d
        print(f"{nombre}: {dt:.1f} s")

    if not HAY_EMBEDKIT:
        print("embedkit no esta instalado; se omite el refinamiento.")
        return pd.DataFrame(filas).T, embeddings

    def refinar(mode, target_dim, y_fit=None):
        t0 = time.time()
        X_ref = EmbedKit(
            mode=mode, target_dim=target_dim, epochs=epochs, random_state=seed
        ).fit_transform(X.astype("float32"), y=y_fit)
        dt = time.time() - t0
        return X_ref, dt

    corridas = [("EmbedKit self-sup (2D)", "self_supervised", 2, None)]
    if con_supervisado:
        corridas.append(("EmbedKit self-sup (3D)", "self_supervised", 3, None))
        corridas.append(("EmbedKit superv. (3D)", "supervised", 3, y))
    else:
        corridas.append(("EmbedKit self-sup (3D)", "self_supervised", 3, None))

    for nombre, mode, dim, y_fit in corridas:
        X_ref, dt = refinar(mode, dim, y_fit)
        metricas = evaluar(X, X_ref, y, k)
        metricas["Tiempo (s)"] = dt
        filas[nombre] = metricas
        embeddings[nombre] = X_ref
        print(f"{nombre}: refinado a R^{dim}; {dt:.1f} s")

    return pd.DataFrame(filas).T, embeddings


# ---------------------------------------------------------------------------
# 4. Seccion B: generalizacion fuera de muestra (Seccion 8 del notebook)
# ---------------------------------------------------------------------------

def correr_seccion_b_generalizacion(X, y, k, dim_gen, epochs, con_supervisado, seed):
    encs = {"PCA": lambda: PCA(n_components=dim_gen, random_state=seed)}
    if HAY_UMAP:
        encs["UMAP"] = lambda: umap.UMAP(n_components=dim_gen, n_neighbors=k, random_state=seed)

    supervisados = set()
    if HAY_EMBEDKIT:
        encs["EmbedKit self-sup"] = lambda: EmbedKit(
            mode="self_supervised", target_dim=dim_gen, epochs=epochs, random_state=seed
        )
        if con_supervisado:
            encs["EmbedKit superv."] = lambda: EmbedKit(
                mode="supervised", target_dim=dim_gen, epochs=epochs, random_state=seed
            )
            supervisados.add("EmbedKit superv.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    resultados = {nombre: {"acc": [], "tw": []} for nombre in encs}

    for tr, te in skf.split(X, y):
        for nombre, fabrica in encs.items():
            modelo = fabrica()
            if nombre in supervisados:
                Z_tr = modelo.fit_transform(X[tr].astype("float32"), y=y[tr])
                Z_te = modelo.transform(X[te].astype("float32"))
            elif nombre.startswith("EmbedKit"):
                Z_tr = modelo.fit_transform(X[tr].astype("float32"))
                Z_te = modelo.transform(X[te].astype("float32"))
            else:
                Z_tr = modelo.fit_transform(X[tr])
                Z_te = modelo.transform(X[te])

            knn = KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y[tr])
            resultados[nombre]["acc"].append(knn.score(Z_te, y[te]))
            resultados[nombre]["tw"].append(trustworthiness(X[te], Z_te, n_neighbors=k))

    base_accs = [
        KNeighborsClassifier(n_neighbors=k).fit(X[tr], y[tr]).score(X[te], y[te])
        for tr, te in skf.split(X, y)
    ]

    filas = {}
    for nombre, r in resultados.items():
        filas[nombre] = {
            "OOS kNN acc (media)": np.mean(r["acc"]),
            "OOS kNN acc (std)": np.std(r["acc"]),
            "OOS trustworthiness (media)": np.mean(r["tw"]),
        }
        print(f"{nombre}   OOS kNN acc = {np.mean(r['acc']):.3f} +/- {np.std(r['acc']):.3f}")
    filas["Base (R^784, sin embeber)"] = {
        "OOS kNN acc (media)": np.mean(base_accs),
        "OOS kNN acc (std)": np.std(base_accs),
        "OOS trustworthiness (media)": np.nan,
    }

    return pd.DataFrame(filas).T


# ---------------------------------------------------------------------------
# 5. Graficas
# ---------------------------------------------------------------------------

def graficar_embeddings_2d(embeddings, y, outdir):
    import matplotlib.pyplot as plt

    nombres = list(embeddings.keys())
    n = len(nombres)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, nombre in zip(axes, nombres):
        Z = embeddings[nombre]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=6, alpha=0.7)
        ax.set_title(nombre)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    handles, _ = sc.legend_elements()
    fig.legend(handles, NOMBRES_CLASES, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Embeddings 2D en Fashion-MNIST (coloreado por clase)", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    ruta = outdir / "fig_embeddings_2d.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Guardado: {ruta}")


def graficar_metricas_barras(tabla, outdir):
    import matplotlib.pyplot as plt

    cols = ["Trustworthiness", "Continuity", "kNN acc", "Silhouette", "Corr. distancias"]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4.5))
    for ax, col in zip(axes, cols):
        tabla[col].plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=75)
    fig.suptitle("Comparacion de metricas por tecnica (Fashion-MNIST)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ruta = outdir / "fig_metricas_barras.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Guardado: {ruta}")


def graficar_generalizacion(tabla_gen, outdir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    tabla_gen["OOS kNN acc (media)"].plot(
        kind="bar", yerr=tabla_gen["OOS kNN acc (std)"], ax=ax, color="darkorange", capsize=4
    )
    ax.set_ylabel("Exactitud k-NN fuera de muestra")
    ax.set_title("Generalizacion fuera de muestra (5-fold, Fashion-MNIST)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    ruta = outdir / "fig_generalizacion.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Guardado: {ruta}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000, help="tamano de la submuestra estratificada")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10, help="numero de vecinos comun")
    ap.add_argument("--epochs", type=int, default=150, help="epocas de entrenamiento de EmbedKit")
    ap.add_argument("--dim-gen", type=int, default=3, help="dimension para la prueba de generalizacion")
    ap.add_argument("--outdir", type=str, default="resultados_fashion_mnist")
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

    print("=" * 70)
    print("Experimento: UMAP vs EmbedKit (no supervisado) en Fashion-MNIST")
    print("=" * 70)

    X_full, y_full = cargar_fashion_mnist()
    print(f"Fashion-MNIST completo: {X_full.shape}, clases: {sorted(np.unique(y_full))}")

    X, y = submuestra_estratificada(X_full, y_full, args.n, args.seed)

    if HAY_EMBEDKIT:
        print("\n--- Diagnostico EmbedKitAnalyzer sobre datos crudos ---")
        reporte = EmbedKitAnalyzer(k=args.k).fit(X.astype("float32"))
        reporte.print_summary()

    print("\n--- Seccion A: embeddings 2D/3D y metricas ---")
    tabla_a, embeddings = correr_seccion_a(
        X, y, args.k, args.epochs, args.con_tsne, con_supervisado, args.seed
    )
    print(tabla_a)
    tabla_a.to_csv(outdir / "tabla_metricas_2d.csv")

    print("\n--- Seccion B: generalizacion fuera de muestra ---")
    tabla_b = correr_seccion_b_generalizacion(
        X, y, args.k, args.dim_gen, args.epochs, con_supervisado, args.seed
    )
    print(tabla_b)
    tabla_b.to_csv(outdir / "generalizacion_oos.csv")

    print("\n--- Graficas ---")
    graficar_embeddings_2d(
        {n: e for n, e in embeddings.items() if e.shape[1] >= 2}, y, outdir
    )
    graficar_metricas_barras(tabla_a, outdir)
    graficar_generalizacion(tabla_b, outdir)

    config = vars(args) | {"hay_umap": HAY_UMAP, "hay_embedkit": HAY_EMBEDKIT}
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nListo. Resultados en: {outdir.resolve()}")


if __name__ == "__main__":
    main()
