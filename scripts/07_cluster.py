"""Clustering non supervisé (K-means, GMM, HDBSCAN) et validation a posteriori avec les étiquettes du catalogue."""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from stars import config, plots, spectra
from stars.training import write_json


def external_scores(labels, cat):
    """ARI et NMI contre chaque étiquette du catalogue, plus la pureté des amas ouverts OCCAM."""
    out = {}
    for col in ["class", "population"]:
        known = (cat[col] != "").to_numpy()
        out[f"ari_{col}"] = float(adjusted_rand_score(cat[col][known], labels[known]))
        out[f"nmi_{col}"] = float(normalized_mutual_info_score(cat[col][known], labels[known]))

    # Les étoiles d'un même amas ouvert partagent âge et chimie : un bon clustering les regroupe.
    members = cat["occam_cluster"].to_numpy()
    purities = []
    for name in set(members) - {""}:
        in_cluster = labels[members == name]
        if len(in_cluster) >= 5:
            purities.append(np.bincount(in_cluster[in_cluster >= 0]).max(initial=0) / len(in_cluster))
    out["occam_purity"] = float(np.mean(purities)) if purities else None
    out["occam_clusters_used"] = len(purities)
    return out


def scan_kmeans(Z, ks, seeds, rng):
    """Pour chaque k : inertie, silhouette et stabilité (ARI moyen entre graines différentes)."""
    sil_idx = rng.choice(len(Z), size=min(5000, len(Z)), replace=False)
    rows = []
    for k in ks:
        runs = [KMeans(k, n_init=1, random_state=s).fit(Z) for s in range(seeds)]
        labels = [r.labels_ for r in runs]
        stability = np.mean([adjusted_rand_score(a, b) for a, b in itertools.combinations(labels, 2)])
        best = min(runs, key=lambda r: r.inertia_)
        rows.append({
            "k": k,
            "inertia": best.inertia_,
            "silhouette": silhouette_score(Z[sil_idx], best.labels_[sil_idx]),
            "stability_ari": stability,
            "gmm_bic": GaussianMixture(k, covariance_type="full", random_state=config.SEED).fit(Z).bic(Z),
        })
        print(f"  k={k:2d}  silhouette {rows[-1]['silhouette']:.3f}  stabilité {stability:.3f}", flush=True)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", choices=["pca", "latent"], default="pca")
    p.add_argument("--ae-run", default="autoencoder_none_good", help="run d'autoencodeur pour --features latent")
    p.add_argument("--subset", choices=["good", "all"], default="good")
    p.add_argument("--n-dims", type=int, default=10, help="dimensions gardées (PCA seulement)")
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--k", type=int, help="k final (par défaut : meilleure silhouette)")
    p.add_argument("--seeds", type=int, default=10, help="graines K-means pour mesurer la stabilité")
    p.add_argument("--min-cluster-size", type=int, default=100, help="paramètre HDBSCAN")
    p.add_argument("--hdbscan-n", type=int, default=50000, help="sous-échantillon pour HDBSCAN, trop lent sur 250 000 étoiles")
    args = p.parse_args()

    _, cat = spectra.load_dataset(with_spectra=False)
    if args.features == "pca":
        Z = np.load(config.PCA_FILE)["Z"][:, :args.n_dims]
    else:
        Z = np.load(config.RUNS / args.ae_run / "latent.npy")
    keep = cat["quality"].eq("good").to_numpy() if args.subset == "good" else np.ones(len(cat), bool)
    Z, cat = StandardScaler().fit_transform(Z[keep]), cat[keep].reset_index(drop=True)

    tag = f"{args.features}_{args.subset}" + (f"_k{args.k}" if args.k else "")
    out = config.RUNS / f"cluster_{tag}"
    fig_dir = config.FIGURES / "cluster" / tag
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)
    print(f"{len(Z)} étoiles, {Z.shape[1]} dimensions")

    scan = scan_kmeans(Z, range(args.k_min, args.k_max + 1), args.seeds, rng)
    scan.to_csv(out / "kmeans_scan.csv", index=False)
    plots.save_line_plot(scan.k, [scan.silhouette, scan.stability_ari], ["silhouette", "stabilité (ARI)"],
                         fig_dir / "kmeans_scan.png", "k", "score", "Choix de k")
    k = args.k or int(scan.loc[scan.silhouette.idxmax(), "k"])
    print(f"k retenu : {k} (BIC GMM minimal à k={int(scan.loc[scan.gmm_bic.idxmin(), 'k'])})")

    sub = np.sort(rng.choice(len(Z), size=min(args.hdbscan_n, len(Z)), replace=False))
    labelings = {
        "kmeans": (KMeans(k, n_init=10, random_state=config.SEED).fit_predict(Z), cat),
        "gmm": (GaussianMixture(k, covariance_type="full", random_state=config.SEED).fit(Z).predict(Z), cat),
        "hdbscan": (HDBSCAN(min_cluster_size=args.min_cluster_size, copy=True).fit_predict(Z[sub]),
                    cat.iloc[sub].reset_index(drop=True)),
    }

    results = {**vars(args), "k": k, "n_stars": len(Z)}
    for method, (labels, cat) in labelings.items():
        scores = external_scores(labels, cat)
        scores["n_scored"] = len(labels)
        scores["n_clusters"] = int(len(set(labels) - {-1}))
        scores["noise_frac"] = float(np.mean(labels == -1))
        results[method] = scores
        print(method, scores)
        plots.kiel(cat, labels, fig_dir / f"kiel_{method}.png", f"{method} (k={scores['n_clusters']})")
        pd.crosstab(labels, cat["class"].replace("", "?")).to_csv(out / f"crosstab_{method}_class.csv")

    write_json(out / "metrics.json", results)
    print(f"Résultats dans {out}, figures dans {fig_dir}")


if __name__ == "__main__":
    main()
