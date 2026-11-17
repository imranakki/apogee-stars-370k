"""Figures du rapport, de l'affiche et de la présentation, à partir des résultats de l'expérience complète."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from stars import config, spectra, tasks
from stars.models import SpectrumCNN
from stars.training import predict

OUT = Path(__file__).resolve().parent / "figures"
INK = "#14202B"
# Palette Okabe-Ito : teintes et luminosités éloignées, distinctes pour les daltoniens et en niveaux de gris.
BLUE, GREEN, ORANGE, VERMIL, BLACK = "#0072B2", "#009E73", "#E69F00", "#D55E00", "#000000"
CLASSES = {"dwarf": ("Naines", BLUE), "subgiant": ("Sous-géantes", GREEN), "rgb": ("RGB", ORANGE), "rc": ("Red clump", BLACK)}
# Ordre de tracé : les grandes classes d'abord, les petites par-dessus pour rester visibles.
DRAW_ORDER = ["dwarf", "rgb", "subgiant", "rc"]

plt.rcParams.update({
    "font.size": 22, "axes.titlesize": 26, "axes.labelsize": 24, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.5,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def tsne_figure(Z, cat):
    idx = np.random.default_rng(config.SEED).choice(len(Z), 10000, replace=False)
    emb = TSNE(perplexity=30, init="pca", random_state=config.SEED).fit_transform(Z[idx])
    sub = cat.iloc[idx]
    quality = {"good": ("Bons spectres", BLUE), "bad": ("Mauvais spectres", VERMIL)}
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4))
    by_class = {k: CLASSES[k] for k in DRAW_ORDER}
    for ax, col, palette, title in [(axes[0], "class", by_class, "Coloré par classe (catalogue)"),
                                    (axes[1], "quality", quality, "Coloré par qualité")]:
        values = sub[col].to_numpy()
        for key, (label, color) in palette.items():
            m = values == key
            ax.scatter(*emb[m].T, s=7, c=color, label=label, rasterized=True, alpha=0.85, linewidths=0)
        ax.set_title(title, loc="left", pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(markerscale=5, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.02), ncol=2, fontsize=20)
    save(fig, "tsne.png")


def kiel_figure(Z, cat):
    good = cat.quality.eq("good").to_numpy()
    Zg = StandardScaler().fit_transform(Z[:, :10][good])
    labels = GaussianMixture(3, covariance_type="full", random_state=config.SEED).fit(Zg).predict(Zg)
    cg = cat[good]
    # Le GMM numérote ses groupes au hasard : chaque nom suit le contenu du groupe, pas son numéro.
    counts = pd.crosstab(labels, cg["class"])
    dwarf_group = counts["dwarf"].idxmax()
    rc_group = counts.drop(dwarf_group)["rc"].idxmax()
    rgb_group = next(k for k in counts.index if k not in (dwarf_group, rc_group))
    groups = {dwarf_group: ("Groupe A : séquence principale", BLUE), rgb_group: ("Groupe B : haut de la RGB", ORANGE),
              rc_group: ("Groupe C : red clump + bas RGB", BLACK)}
    fig, ax = plt.subplots(figsize=(10, 8.4))
    for k, (name, color) in groups.items():
        m = labels == k
        ax.scatter(cg.TEFF[m], cg.LOGG[m], s=5, c=color, label=name, rasterized=True, alpha=0.8, linewidths=0)
    ax.set_xlim(7000, 3000)
    ax.set_ylim(5.4, -0.4)
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("log g")
    ax.legend(markerscale=7, frameon=False, loc="upper left", fontsize=19)
    save(fig, "kiel_gmm.png")
    print(pd.crosstab(labels, cg["class"]))


def snr_figure():
    q = pd.read_csv(config.DATA / "quality_by_snr.csv")
    bins, ticks = ["5-30", "30-70", "70-150", "150-inf"], ["5–30", "30–70", "70–150", "> 150"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
    for ax, metric, unit, title in [(axes[0], "mae_LOGG", "dex", "Erreur sur log g"), (axes[1], "mae_TEFF", "K", "Erreur sur Teff")]:
        for run, label, color, marker, style in [("cnn_regress_good", "Entraîné sur les bons", BLUE, "o", "-"),
                                                 ("cnn_regress_all", "Entraîné sur tous", VERMIL, "s", "--")]:
            y = q[q.run == run].set_index("snr_bin").reindex(bins)[metric].to_numpy()
            ax.plot(range(4), y, marker=marker, ms=12, lw=3.5, ls=style, c=color, label=label)
        ax.set_xticks(range(4), ticks)
        ax.set_xlabel("SNR du spectre")
        ax.set_ylabel(f"erreur absolue moyenne ({unit})")
        ax.set_title(title, loc="left", pad=12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, None)
    axes[0].legend(frameon=False, fontsize=20)
    save(fig, "snr.png")


def kmeans_scan_figure():
    scan = pd.read_csv(config.RUNS / "cluster_pca_good" / "kmeans_scan.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(scan.k, scan.stability_ari, marker="o", ms=10, lw=3, ls="--", c=VERMIL, label="stabilité (ARI entre 10 graines)")
    ax.plot(scan.k, scan.silhouette, marker="s", ms=10, lw=3, c=BLUE, label="silhouette")
    ax.set_xlabel("nombre de groupes k")
    ax.set_ylabel("score")
    ax.set_xticks(scan.k)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=19, loc="center right")
    save(fig, "kmeans_scan.png")


def mean_spectra_figure(X, cat):
    wave = spectra.wavelengths(np.load(config.FEATURES)["keep_pixels"])
    window = (wave > 15700) & (wave < 15800)
    good = cat.quality.eq("good").to_numpy()
    Xw = X[:, window]
    fig, ax = plt.subplots(figsize=(14, 7.5))
    # Courbes décalées verticalement : les quatre spectres ne se superposent plus.
    for i, (key, (label, color)) in enumerate(CLASSES.items()):
        m = good & (cat["class"] == key).to_numpy()
        offset = -0.35 * i
        ax.plot(wave[window], Xw[m].mean(axis=0, dtype=np.float64) + offset, lw=2.5, c=color)
        ax.text(15802, 1.03 + offset, f"{label}\n({m.sum()})", color=color, fontsize=18, va="center", fontweight="bold")
    ax.set_xlim(15700, 15830)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("longueur d'onde (Å)")
    ax.set_ylabel("flux normalisé, décalé par classe")
    save(fig, "mean_spectra_zoom.png")


def confusion_figure(X, cat):
    run = config.RUNS / "cnn_class_good"
    cfg = json.loads((run / "config.json").read_text())
    masks = tasks.split_masks(cat, "class", cfg["train_on"])
    y = tasks.targets(cat, "class")
    model = SpectrumCNN(len(config.CLASSES))
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=False)["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    m = masks["test_good"]
    pred = predict(model, X, device, np.flatnonzero(m)).argmax(axis=1)
    cm = confusion_matrix(y[m], pred, labels=range(len(config.CLASSES)))
    rates = cm / cm.sum(axis=1, keepdims=True)
    names = [CLASSES[c][0] for c in config.CLASSES]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{rates[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=18,
                    color="white" if rates[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(names)), names, rotation=20)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("classe prédite")
    ax.set_ylabel("classe du catalogue")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "confusion_cnn.png")
    print(cm)


def main():
    OUT.mkdir(exist_ok=True)
    X, cat = spectra.load_dataset()
    Z = np.load(config.PCA_FILE)["Z"]
    kmeans_scan_figure()
    snr_figure()
    mean_spectra_figure(X, cat)
    confusion_figure(X, cat)
    kiel_figure(Z, cat)
    tsne_figure(Z, cat)
    print(f"Figures dans {OUT}")


if __name__ == "__main__":
    main()
