import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# Palette « bright » de Paul Tol : distincte pour les daltoniens et en niveaux de gris.
plt.rcParams["axes.prop_cycle"] = cycler(color=["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"])


def _panel(ax, xy, values, name, ticks=False):
    if values.dtype.kind in "fi":
        ok = np.isfinite(values)
        ax.scatter(*xy[~ok].T, s=1, c="lightgrey", rasterized=True)
        lo, hi = np.percentile(values[ok], [2, 98])
        sc = ax.scatter(*xy[ok].T, s=1, c=values[ok], cmap="viridis", vmin=lo, vmax=hi, rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.8)
    else:
        labels = np.where(values == "", "?", values)
        for label in sorted(set(labels), key=lambda s: (s == "?", s)):
            m = labels == label
            color = "lightgrey" if label == "?" else None
            ax.scatter(*xy[m].T, s=1, label=f"{label} ({m.sum()})", c=color, rasterized=True)
        ax.legend(markerscale=8, fontsize=7, loc="best")
    ax.set_title(name)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def embedding_panels(xy, cat, columns, path, title):
    """Un panneau par colonne du catalogue : xy coloré par cette colonne."""
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4.5))
    for ax, col in zip(np.atleast_1d(axes), columns):
        values = cat[col].to_numpy()
        _panel(ax, xy, values.astype(float) if values.dtype.kind in "fi" else values.astype(str), col)
    fig.suptitle(title)
    _save(fig, path)


def kiel(cat, labels, path, title):
    """Diagramme de Kiel (Teff vs log g), couleur = labels. Les axes sont inversés comme en astrophysique."""
    fig, ax = plt.subplots(figsize=(6, 5))
    xy = cat[["TEFF", "LOGG"]].to_numpy(dtype=float)
    _panel(ax, xy, np.asarray(labels).astype(str), title, ticks=True)
    ax.set_xlim(7000, 3000)
    ax.set_ylim(5.5, -0.5)
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("log g")
    _save(fig, path)


def save_line_plot(x, ys, labels, path, xlabel, ylabel, title, xticklabels=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    for y, label in zip(ys, labels):
        ax.plot(x, y, lw=0.6 if xticklabels is None else 2, marker=None if xticklabels is None else "o", label=label)
    if xticklabels is not None:
        ax.set_xticks(x, xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(ys) > 1:
        ax.legend(fontsize=8)
    _save(fig, path)
