"""Figures des tests de sensibilité (scripts/11_sensitivity.py), dans report/figures/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stars import config

SRC = config.DATA / "sensitivity"
OUT = Path(__file__).resolve().parent / "figures"
BLUE, VERMIL, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#E69F00", "#009E73", "#6B7480"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    path = SRC / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def ref_line(ax, x, label="référence"):
    ax.axvline(x, color=GREY, ls=":", lw=1)
    ax.text(x, ax.get_ylim()[1], f" {label}", color=GREY, fontsize=9, va="top")


def two_metrics(axes, df, x, xlabel, ref, logx=False, title=""):
    """Macro F1 (classification) à gauche, erreur sur Teff (régression) à droite, sur test_good."""
    c = df[df.task == "class"].sort_values(x)
    r = df[df.task == "regress"].sort_values(x)
    axes[0].plot(c[x], c["test_good.macro_f1"], "o-", color=BLUE)
    axes[0].set_ylabel("macro F1 (test_good)")
    axes[1].plot(r[x], r["test_good.mae_TEFF"], "s-", color=VERMIL)
    axes[1].set_ylabel("erreur Teff (K, test_good)")
    for ax in axes:
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ref_line(ax, ref)
    axes[0].set_title(title, loc="left", fontsize=11)


def data_figure():
    tests = [("snr", "seuil de SNR des bons spectres", 70, False, "Seuil de qualité"),
             ("badfrac", "part des mauvais spectres ajoutés", 0.0, False, "Mauvais spectres"),
             ("size", "part des bons spectres d'entraînement", 1.0, True, "Taille d'entraînement (CNN)")]
    tests = [t for t in tests if load(t[0]) is not None]
    if not tests:
        return
    fig, axes = plt.subplots(2, len(tests), figsize=(4.6 * len(tests), 7), squeeze=False)
    for j, (name, xlabel, ref, logx, title) in enumerate(tests):
        df = load(name)
        if name == "size":
            rf = df[df.model == "rf"]
            df = df[df.model == "cnn"]
        two_metrics(axes[:, j], df.rename(columns={"value": "x"}), "x", xlabel, ref, logx, title)
        if name == "size":
            c, r = rf[rf.task == "class"].sort_values("value"), rf[rf.task == "regress"].sort_values("value")
            axes[0, j].plot(c.value, c["test_good.macro_f1"], "o--", color=GREY, label="forêt")
            axes[1, j].plot(r.value, r["test_good.mae_TEFF"], "s--", color=GREY, label="forêt")
            axes[0, j].legend(frameon=False)
        if name == "badfrac":
            r = df[df.task == "regress"].sort_values("value")
            axes[1, j].plot(r.value, r["test_low.mae_TEFF"] / 4, "s--", color=ORANGE, label="SNR < 70, erreur ÷ 4")
            axes[1, j].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "sens_data.png", dpi=200)
    plt.close(fig)


def hyper_figure():
    panels = []
    for task, ylabel, col in [("class", "macro F1 (test_good)", "test_good.macro_f1"), ("regress", "erreur Teff (K)", "test_good.mae_TEFF")]:
        df = load(f"epochs_{task}")
        if df is not None:
            panels.append(("epochs", task, ylabel, col, df))
    for name in ["lr", "rf_trees", "rf_pca"]:
        df = load(name)
        if df is not None:
            panels.append((name, None, None, None, df))
    if not panels:
        return
    ncol = 3
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 4 * nrow), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (name, task, ylabel, col, df) in zip(axes.flat, panels):
        if name == "epochs":
            best = df.loc[df.val_loss.idxmin(), "epoch"]
            ax.plot(df.epoch, df[col], "-", color=BLUE if task == "class" else VERMIL)
            ax.axvline(best, color=GREY, ls="--", lw=1)
            ax.axvline(30, color=GREY, ls=":", lw=1)
            ax.set_xlabel(f"époques (tirets : meilleure validation, {int(best)}, pointillés : 30)")
            ax.set(ylabel=ylabel, title="CNN, " + ("classes" if task == "class" else "Teff"))
        elif name == "lr":
            ax.plot(df.value, df["test_good.macro_f1"], "o-", color=BLUE)
            ax.set_xscale("log")
            ref_line(ax, 1e-3)
            ax.set(xlabel="taux d'apprentissage", ylabel="macro F1 (test_good)", title="CNN, taux d'apprentissage")
        else:
            c, r = df[df.task == "class"].sort_values("value"), df[df.task == "regress"].sort_values("value")
            ax.plot(c.value, c["test_good.macro_f1"], "o-", color=BLUE, label="macro F1")
            ax.set_ylabel("macro F1 (test_good)", color=BLUE)
            ax2 = ax.twinx()
            ax2.plot(r.value, r["test_good.mae_TEFF"], "s--", color=VERMIL, label="erreur Teff")
            ax2.set_ylabel("erreur Teff (K)", color=VERMIL)
            ax2.spines["top"].set_visible(False)
            ax.set_xscale("log")
            ref_line(ax, 300 if name == "rf_trees" else 50)
            ax.set(xlabel="arbres" if name == "rf_trees" else "composantes PCA",
                   title="Forêt, " + ("nombre d'arbres" if name == "rf_trees" else "composantes PCA"))
    fig.tight_layout()
    fig.savefig(OUT / "sens_hyper.png", dpi=200)
    plt.close(fig)


def cluster_figure():
    dims, ks = load("cluster_dims"), load("cluster_k")
    if dims is None and ks is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    if dims is not None:
        for (method, k), g in dims.groupby(["method", "k"]):
            g = g.sort_values("dims")
            axes[0].plot(g.dims, g.ari, "o-" if method == "gmm" else "s--",
                         color=BLUE if method == "gmm" else ORANGE, alpha=1 if k == 3 else 0.5,
                         label=f"{'GMM' if method == 'gmm' else 'K-means'}, k = {k}")
        axes[0].set_xscale("log")
        axes[0].set_xticks([2, 3, 5, 10, 20, 30, 50])
        axes[0].set_xticklabels(["2", "3", "5", "10", "20", "30", "50"])
        axes[0].minorticks_off()
        ref_line(axes[0], 10)
        axes[0].set(xlabel="composantes PCA", ylabel="ARI contre les classes", title="Nombre de composantes")
        axes[0].legend(frameon=False, fontsize=9)
    if ks is not None:
        for method, color, marker in [("gmm", BLUE, "o"), ("kmeans", ORANGE, "s")]:
            g = ks.groupby("k")[f"{method}_ari"].agg(["median", "min", "max"]).reset_index()
            axes[1].plot(g.k, g["median"], marker + "-", color=color, label=f"{'GMM' if method == 'gmm' else 'K-means'}, médiane")
            axes[1].fill_between(g.k, g["min"], g["max"], color=color, alpha=0.2)
        axes[1].set(xlabel="nombre de groupes k", ylabel="ARI contre les classes", title="10 graines par k (bande : min à max)")
        axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "sens_cluster.png", dpi=200)
    plt.close(fig)


def best_figure():
    """Référence contre meilleurs réglages, sur test_good. CNN : moyenne et écart type de 3 graines."""
    path = config.DATA / "best" / "final.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    panels = [("class", "test_good.macro_f1", "macro F1"), ("class", "test_low.macro_f1", "macro F1, SNR < 70"),
              ("regress", "test_good.mae_LOGG", "erreur log g"), ("regress", "test_good.mae_M_H", "erreur [M/H]")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (task, col, title) in zip(axes, panels):
        g = df[df.task == task].groupby(["model", "setting"])[col].agg(["mean", "std"]).fillna(0)
        x = 0
        for model, label in [("cnn", "CNN"), ("rf", "Forêt")]:
            for setting, color in [("reference", GREY), ("best", BLUE)]:
                m, s = g.loc[(model, setting)]
                ax.bar(x, m, yerr=s, color=color, capsize=4, width=0.8)
                ax.text(x, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
                x += 1
            x += 0.6
        ax.set_xticks([0.5, 3.1])
        ax.set_xticklabels(["CNN", "Forêt"])
        lo = g["mean"].min()
        ax.set_ylim(lo * 0.9 if task == "class" else 0, None)
        ax.set_title(title, fontsize=11)
    axes[0].bar(0, 0, color=GREY, label="référence")
    axes[0].bar(0, 0, color=BLUE, label="meilleurs réglages")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "sens_best.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    best_figure()
    data_figure()
    hyper_figure()
    cluster_figure()
    print(f"Figures dans {OUT}")
