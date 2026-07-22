"""
Barrido de configuraciones de refinamiento NO SUPERVISADO de EmbedKit en Fashion-MNIST.

Objetivo: encontrar una configuracion del refinador contrastivo self-supervised
de embedding-kit que **supere a UMAP** como embedder no supervisado.

Contexto (de experimentos previos en este repo): a target_dim=16 EmbedKit self-sup
ya iguala o supera a UMAP en metricas *dentro de muestra* (trustworthiness, kNN acc),
pero pierde claramente en **generalizacion fuera de muestra** y en **estabilidad**
(su exactitud OOS colapsa en algunos folds). Por eso este barrido optimiza sobre
todo la **exactitud k-NN fuera de muestra** (OOS), que es donde esta la brecha real.

EmbedKit expone varios "mecanismos" de refinamiento, todos combinables en modo
self-supervised (sin usar etiquetas nunca):
  - Augmentation (define los pares positivos): KNNPairs, EmbeddingMixup,
    GaussianNoise, FeatureDropout, FeatureMasking, CompositeAugmentation.
  - Loss: NTXentLoss (InfoNCE, con temperatura), AlignUniformLoss, y CombinedLoss.
    (TripletLoss / SupConLoss / RankNContrastLoss requieren etiquetas -> excluidos.)
  - Arquitectura del proyector: target_dim, hidden_dim, n_layers.
  - Entrenamiento: epochs, lr, batch_size.

Estrategia: descenso por coordenadas (coordinate descent) por etapas —
  Etapa 1: dimension objetivo (target_dim).
  Etapa 2: augmentation (con la mejor dimension).
  Etapa 3: loss / temperatura.
  Etapa 4: arquitectura y entrenamiento.
  Etapa 5 (finales): la mejor config vs. la config por defecto vs. UMAP, con
    validacion 5-fold y varias semillas -> media +/- std de la exactitud OOS.

Durante el barrido cada config se evalua con UN holdout estratificado (rapido);
los finalistas se re-evaluan con 5-fold x varias semillas (robusto).

Uso:
    python embedkit_unsup_sweep_fashion_mnist.py [--n 3000] [--seed 0] [--k 10]
        [--epochs 150] [--outdir resultados_sweep] [--quick]
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# Algunas versiones de umap-learn importan TensorFlow de forma ansiosa si esta
# instalado, y TF + PyTorch (embedding-kit) en el mismo interprete provocan un
# Segmentation fault. Bloqueamos `tensorflow` antes de importar umap; umap
# detecta el ImportError y sigue con su UMAP normal. (Este barrido nunca usa TF.)
sys.modules.setdefault("tensorflow", None)

import umap
from embedkit import EmbedKit
from embedkit.improvement.augmentation import (
    CompositeAugmentation,
    EmbeddingMixup,
    FeatureMasking,
    GaussianNoise,
    KNNPairs,
)
from embedkit.improvement.losses import AlignUniformLoss, CombinedLoss, NTXentLoss

# Reutiliza la carga de datos y la funcion de metricas del experimento base.
from fashion_mnist_umap_vs_embedkit import (
    cargar_fashion_mnist,
    evaluar,
    submuestra_estratificada,
)


# ---------------------------------------------------------------------------
# Construccion de configuraciones
# ---------------------------------------------------------------------------
# Una config es un dict. `aug` y `loss` se guardan como (etiqueta, builder) para
# construir SIEMPRE un objeto nuevo por ajuste (KNNPairs y otras guardan estado).

def loss_default():
    return CombinedLoss([(NTXentLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.5)])


def cfg_base(target_dim=16, epochs=150):
    return {
        "target_dim": target_dim,
        "aug": ("KNNPairs(k=5)", lambda: KNNPairs(k=5)),
        "loss": ("Combined(NTXent.07 + AU.5)", loss_default),
        "hidden_dim": 256,
        "n_layers": 2,
        "epochs": epochs,
        "lr": 3e-4,
        "batch_size": 256,
    }


def make_embedkit(cfg, seed):
    return EmbedKit(
        mode="self_supervised",
        augmentation=cfg["aug"][1](),
        loss=cfg["loss"][1](),
        target_dim=cfg["target_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        random_state=seed,
    )


def describe(cfg):
    return (
        f"dim={cfg['target_dim']} | aug={cfg['aug'][0]} | loss={cfg['loss'][0]} | "
        f"h={cfg['hidden_dim']} L={cfg['n_layers']} | lr={cfg['lr']} ep={cfg['epochs']}"
    )


# ---------------------------------------------------------------------------
# Evaluacion rapida (un holdout) para el barrido
# ---------------------------------------------------------------------------

def eval_holdout_embedkit(cfg, X, y, k, seed, test_size=0.25):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (tr, te), = sss.split(X, y)
    enc = make_embedkit(cfg, seed)
    t0 = time.time()
    Z_tr = enc.fit_transform(X[tr].astype("float32"))
    dt = time.time() - t0
    Z_te = enc.transform(X[te].astype("float32"))
    knn = KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y[tr])
    return {
        "OOS kNN acc": knn.score(Z_te, y[te]),
        "OOS trust": trustworthiness(X[te], Z_te, n_neighbors=k),
        "train trust": trustworthiness(X[tr], Z_tr, n_neighbors=k),
        "Tiempo (s)": dt,
    }


def eval_holdout_umap(dim, X, y, k, seed, test_size=0.25):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (tr, te), = sss.split(X, y)
    enc = umap.UMAP(n_components=dim, n_neighbors=k, random_state=seed)
    t0 = time.time()
    Z_tr = enc.fit_transform(X[tr])
    dt = time.time() - t0
    Z_te = enc.transform(X[te])
    knn = KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y[tr])
    return {
        "OOS kNN acc": knn.score(Z_te, y[te]),
        "OOS trust": trustworthiness(X[te], Z_te, n_neighbors=k),
        "train trust": trustworthiness(X[tr], Z_tr, n_neighbors=k),
        "Tiempo (s)": dt,
    }


def correr_etapa(nombre, configs, X, y, k, seed, umap_ref=None):
    print(f"\n{'='*70}\nEtapa: {nombre}\n{'='*70}")
    filas = {}
    for etiqueta, cfg in configs:
        r = eval_holdout_embedkit(cfg, X, y, k, seed)
        filas[etiqueta] = r
        print(f"  {etiqueta:38s} OOS acc={r['OOS kNN acc']:.3f}  "
              f"OOS trust={r['OOS trust']:.3f}  ({r['Tiempo (s)']:.0f}s)")
    if umap_ref is not None:
        print(f"  {'--- UMAP (referencia) ---':38s} OOS acc={umap_ref['OOS kNN acc']:.3f}  "
              f"OOS trust={umap_ref['OOS trust']:.3f}")
    df = pd.DataFrame(filas).T
    mejor = df["OOS kNN acc"].astype(float).idxmax()
    print(f"  -> mejor de la etapa: {mejor} (OOS acc={df.loc[mejor,'OOS kNN acc']:.3f})")
    return df, mejor


# ---------------------------------------------------------------------------
# Evaluacion robusta (5-fold x semillas) para los finalistas
# ---------------------------------------------------------------------------

def eval_kfold_embedkit(cfg, X, y, k, seeds, n_splits=5):
    accs, tws = [], []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            enc = make_embedkit(cfg, seed)
            Z_tr = enc.fit_transform(X[tr].astype("float32"))
            Z_te = enc.transform(X[te].astype("float32"))
            knn = KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y[tr])
            accs.append(knn.score(Z_te, y[te]))
            tws.append(trustworthiness(X[te], Z_te, n_neighbors=k))
    return np.array(accs), np.array(tws)


def eval_kfold_umap(dim, X, y, k, seeds, n_splits=5):
    accs, tws = [], []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            enc = umap.UMAP(n_components=dim, n_neighbors=k, random_state=seed)
            Z_tr = enc.fit_transform(X[tr])
            Z_te = enc.transform(X[te])
            knn = KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y[tr])
            accs.append(knn.score(Z_te, y[te]))
            tws.append(trustworthiness(X[te], Z_te, n_neighbors=k))
    return np.array(accs), np.array(tws)


# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------

def graficar_finalistas(resumen, outdir):
    import matplotlib.pyplot as plt

    nombres = list(resumen.keys())
    medias = [resumen[n]["OOS acc media"] for n in nombres]
    stds = [resumen[n]["OOS acc std"] for n in nombres]
    colores = ["darkorange" if "UMAP" in n else "steelblue" for n in nombres]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(nombres, medias, yerr=stds, color=colores, capsize=4)
    ax.set_ylabel("Exactitud k-NN fuera de muestra (5-fold x semillas)")
    ax.set_title("Finalistas: EmbedKit no supervisado (sweep) vs UMAP")
    ax.tick_params(axis="x", rotation=30)
    for i, (m, s) in enumerate(zip(medias, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    ruta = outdir / "fig_finalistas.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Guardado: {ruta}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--outdir", type=str, default="resultados_sweep")
    ap.add_argument("--quick", action="store_true", help="barrido reducido para pruebas")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    k, seed, ep = args.k, args.seed, args.epochs

    print("=" * 70)
    print("Barrido de refinamiento NO SUPERVISADO de EmbedKit vs UMAP (Fashion-MNIST)")
    print("=" * 70)

    X_full, y_full = cargar_fashion_mnist()
    X, y = submuestra_estratificada(X_full, y_full, args.n, seed)

    correr_sweep(X, y, args, outdir)


def correr_sweep(X, y, args, outdir):
    """Corre el barrido completo (etapas 1-5) sobre una representacion X dada.

    Se separo de main() para poder reutilizar exactamente el mismo barrido con
    otra entrada (p. ej. embeddings preentrenados) sin duplicar codigo: solo
    cambia la representacion X; toda la logica del sweep es identica.
    """
    k, seed, ep = args.k, args.seed, args.epochs
    stage_tables = {}

    # ---- Etapa 1: target_dim ----
    dims = [16] if args.quick else [8, 16, 32]
    configs = [(f"dim={d}", cfg_base(target_dim=d, epochs=ep)) for d in dims]
    umap_por_dim = {d: eval_holdout_umap(d, X, y, k, seed) for d in dims}
    # Referencia UMAP: usa la dimension mediana para el print de esta etapa.
    df1, _ = correr_etapa("1) target_dim", configs, X, y, k, seed)
    for d in dims:
        df1.loc[f"UMAP dim={d}", :] = umap_por_dim[d]
    stage_tables["etapa1_target_dim"] = df1
    # Mejor dimension: la de mayor OOS acc de EmbedKit.
    ek_rows = [f"dim={d}" for d in dims]
    best_dim = int(df1.loc[ek_rows, "OOS kNN acc"].astype(float).idxmax().split("=")[1])
    print(f"\n>> Mejor target_dim para EmbedKit: {best_dim}")
    umap_ref = umap_por_dim[best_dim]

    base = cfg_base(target_dim=best_dim, epochs=ep)

    # ---- Etapa 2: augmentation ----
    def with_aug(label, builder):
        c = dict(base)
        c["aug"] = (label, builder)
        return c

    aug_variants = [
        ("KNNPairs(k=5)", lambda: KNNPairs(k=5)),
        ("KNNPairs(k=10)", lambda: KNNPairs(k=10)),
        ("KNNPairs(k=15)", lambda: KNNPairs(k=15)),
        ("KNNPairs(k=10,hardneg)", lambda: KNNPairs(k=10, hard_negatives=True)),
        ("GaussianNoise(adapt)", lambda: GaussianNoise(std=0.1, adaptive=True, k=10)),
        ("FeatureMasking(.15)", lambda: FeatureMasking(mask_ratio=0.15)),
        ("Mixup(k=10,a=.4)", lambda: EmbeddingMixup(k=10, alpha=0.4)),
        ("KNN(10)+Noise", lambda: CompositeAugmentation(
            [KNNPairs(k=10), GaussianNoise(std=0.1, adaptive=True, k=10)], mode="random_choice")),
    ]
    if args.quick:
        aug_variants = aug_variants[:3]
    configs = [(lbl, with_aug(lbl, b)) for lbl, b in aug_variants]
    df2, best_aug_lbl = correr_etapa("2) augmentation", configs, X, y, k, seed, umap_ref)
    stage_tables["etapa2_augmentation"] = df2
    best_aug = dict(configs)[best_aug_lbl]["aug"]
    base = dict(base); base["aug"] = best_aug
    print(f">> Mejor augmentation: {best_aug[0]}")

    # ---- Etapa 3: loss / temperatura ----
    def with_loss(label, builder):
        c = dict(base)
        c["loss"] = (label, builder)
        return c

    loss_variants = [
        ("NTXent(.05)", lambda: NTXentLoss(temperature=0.05)),
        ("NTXent(.07)", lambda: NTXentLoss(temperature=0.07)),
        ("NTXent(.2)", lambda: NTXentLoss(temperature=0.2)),
        ("NTXent(.5)", lambda: NTXentLoss(temperature=0.5)),
        ("AlignUniform", lambda: AlignUniformLoss()),
        ("Combined(NTX.07+AU.5)", loss_default),
        ("Combined(NTX.2+AU1.0)", lambda: CombinedLoss(
            [(NTXentLoss(temperature=0.2), 1.0), (AlignUniformLoss(), 1.0)])),
    ]
    if args.quick:
        loss_variants = loss_variants[:3]
    configs = [(lbl, with_loss(lbl, b)) for lbl, b in loss_variants]
    df3, best_loss_lbl = correr_etapa("3) loss / temperatura", configs, X, y, k, seed, umap_ref)
    stage_tables["etapa3_loss"] = df3
    best_loss = dict(configs)[best_loss_lbl]["loss"]
    base = dict(base); base["loss"] = best_loss
    print(f">> Mejor loss: {best_loss[0]}")

    # ---- Etapa 4: arquitectura y entrenamiento ----
    def variar(label, **kw):
        c = dict(base)
        c.update(kw)
        return (label, c)

    arch_variants = [
        ("base", dict(base)),
        variar("hidden=512", hidden_dim=512),
        variar("n_layers=3", n_layers=3),
        variar("lr=1e-3", lr=1e-3),
        variar("epochs=300", epochs=min(300, ep * 2) if not args.quick else ep),
        variar("h512+L3+lr1e-3", hidden_dim=512, n_layers=3, lr=1e-3),
    ]
    if args.quick:
        arch_variants = arch_variants[:3]
    df4, best_arch_lbl = correr_etapa("4) arquitectura/entrenamiento", arch_variants, X, y, k, seed, umap_ref)
    stage_tables["etapa4_arquitectura"] = df4
    best_cfg = dict(arch_variants)[best_arch_lbl]
    print(f">> Mejor arquitectura: {best_arch_lbl}")
    print(f"\n>>> CONFIG GANADORA DEL BARRIDO:\n    {describe(best_cfg)}")

    # Guarda las tablas de cada etapa.
    for nombre, df in stage_tables.items():
        df.to_csv(outdir / f"{nombre}.csv")

    # ---- Etapa 5: finalistas robustos (5-fold x semillas) ----
    print(f"\n{'='*70}\nEtapa 5: finalistas robustos (5-fold x semillas)\n{'='*70}")
    seeds = [0] if args.quick else [0, 1, 2]
    finalistas = {
        "EmbedKit default": cfg_base(target_dim=best_dim, epochs=ep),
        "EmbedKit sweep-best": best_cfg,
    }
    resumen = {}
    for nombre, cfg in finalistas.items():
        accs, tws = eval_kfold_embedkit(cfg, X, y, k, seeds)
        resumen[nombre] = {
            "OOS acc media": accs.mean(), "OOS acc std": accs.std(),
            "OOS trust media": tws.mean(), "config": describe(cfg),
        }
        print(f"  {nombre:22s} OOS acc = {accs.mean():.3f} +/- {accs.std():.3f}")
    accs_u, tws_u = eval_kfold_umap(best_dim, X, y, k, seeds)
    resumen[f"UMAP (dim={best_dim})"] = {
        "OOS acc media": accs_u.mean(), "OOS acc std": accs_u.std(),
        "OOS trust media": tws_u.mean(), "config": f"UMAP n_components={best_dim}",
    }
    print(f"  {'UMAP':22s} OOS acc = {accs_u.mean():.3f} +/- {accs_u.std():.3f}")

    df_fin = pd.DataFrame(resumen).T
    df_fin.to_csv(outdir / "finalistas.csv")
    graficar_finalistas(resumen, outdir)

    # Veredicto.
    best_ek = max(
        [n for n in resumen if "UMAP" not in n],
        key=lambda n: resumen[n]["OOS acc media"],
    )
    ek_acc = resumen[best_ek]["OOS acc media"]
    u_acc = resumen[f"UMAP (dim={best_dim})"]["OOS acc media"]
    veredicto = (
        f"La mejor config no supervisada de EmbedKit ({best_ek}) logra "
        f"OOS kNN acc = {ek_acc:.3f} vs UMAP = {u_acc:.3f} "
        f"(dim={best_dim}). "
        + ("EmbedKit SUPERA a UMAP." if ek_acc > u_acc else "UMAP sigue por delante.")
    )
    print(f"\n>>> VEREDICTO: {veredicto}")

    with open(outdir / "resumen.json", "w") as f:
        json.dump({
            "config_ganadora": describe(best_cfg),
            "best_dim": best_dim,
            "veredicto": veredicto,
            "finalistas": {n: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                               for kk, vv in r.items()} for n, r in resumen.items()},
            "args": vars(args),
        }, f, indent=2, ensure_ascii=False)

    print(f"\nListo. Resultados en: {outdir.resolve()}")


if __name__ == "__main__":
    main()
