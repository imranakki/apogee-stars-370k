"""PCA, t-SNE et UMAP des spectres. Les figures montrent si les classes se séparent sans étiquettes."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.manifold import TSNE

from stars import config, plots, spectra

COLOR_BY = ["class", "LOGG", "TEFF", "M_H", "quality"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-components", type=int, default=50, help="composantes PCA gardées")
    p.add_argument("--tsne-n", type=int, default=10000, help="taille du sous-échantillon pour t-SNE et UMAP")
    p.add_argument("--perplexity", type=float, default=30)
    args = p.parse_args()

    X, cat = spectra.load_dataset()
    out = config.FIGURES / "explore"
    print(f"{len(X)} spectres, {X.shape[1]} pixels")

    pca = spectra.fit_pca(X, np.arange(len(X)), args.n_components)
    Z = spectra.project(pca, X)
    np.savez(config.PCA_FILE, Z=Z, components=pca.components_, mean=pca.mean_, explained=pca.explained_variance_ratio_)

    k = np.arange(1, args.n_components + 1)
    plots.save_line_plot(k, [np.cumsum(pca.explained_variance_ratio_)], ["cumulée"], out / "pca_variance.png",
                         "composantes", "variance expliquée", "PCA : variance expliquée cumulée")
    print(f"Variance expliquée par 10 composantes : {pca.explained_variance_ratio_[:10].sum():.3f}")

    wave = spectra.wavelengths(np.load(config.FEATURES)["keep_pixels"])
    plots.save_line_plot(wave, pca.components_[:3], ["PC1", "PC2", "PC3"], out / "pca_components.png",
                         "longueur d'onde (Å)", "poids", "Trois premières composantes")
    classes = [c for c in config.CLASSES if (cat["class"] == c).any()]
    rng = np.random.default_rng(config.SEED)
    # Moyenne sur 5 000 étoiles par classe : copier toute la RGB en float32 dépasserait la RAM.
    means = []
    for c in classes:
        rows = np.flatnonzero((cat["class"] == c).to_numpy())
        rows = np.sort(rng.choice(rows, size=min(5000, len(rows)), replace=False))
        means.append(X[rows].astype(np.float32).mean(axis=0))
    plots.save_line_plot(wave, means, classes, out / "mean_spectra.png",
                         "longueur d'onde (Å)", "flux normalisé", "Spectre moyen par classe")

    plots.embedding_panels(Z[:, :2], cat, COLOR_BY, out / "pca_2d.png", "PCA (PC1, PC2)")
    plots.kiel(cat, cat["class"], out / "kiel_labels.png", "Étiquettes du catalogue")

    idx = rng.choice(len(X), size=min(args.tsne_n, len(X)), replace=False)
    sub = cat.iloc[idx]

    tsne = TSNE(perplexity=args.perplexity, init="pca", random_state=config.SEED).fit_transform(Z[idx])
    plots.embedding_panels(tsne, sub, COLOR_BY, out / "tsne.png", f"t-SNE sur {len(idx)} étoiles")

    try:
        import umap
    except ImportError:
        print("umap-learn absent : UMAP ignoré (pip install umap-learn)")
    else:
        emb = umap.UMAP(random_state=config.SEED).fit_transform(Z[idx])
        plots.embedding_panels(emb, sub, COLOR_BY, out / "umap.png", f"UMAP sur {len(idx)} étoiles")

    print(f"Figures dans {out}")


if __name__ == "__main__":
    main()
