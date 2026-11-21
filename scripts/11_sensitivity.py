"""Tests de sensibilité : un paramètre change à la fois, tout le reste garde sa valeur de référence.

Chaque configuration terminée ajoute une ligne à STARS_DATA/sensitivity/<test>.csv. Une relance saute les lignes déjà écrites.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch import nn

from stars import catalog, config, spectra, tasks
from stars.models import SpectrumCNN
from stars.training import Batches, predict, run_epoch

OUT = config.DATA / "sensitivity"
TASKS = ["class", "regress"]

# Valeurs testées. La valeur de référence du projet figure dans chaque liste.
GRID = {
    "snr": [30, 50, 70, 100, 150],
    "badfrac": [0.0, 0.25, 0.5, 1.0],
    "size": [0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "rf_trees": [10, 50, 100, 300, 500],
    "rf_pca": [5, 10, 20, 50, 100],
    "cluster_dims": [2, 3, 5, 10, 20, 30, 50],
    "cluster_k": list(range(2, 9)),
}
EPOCHS_MAX = 50


def done_keys(name):
    path = OUT / f"{name}.csv"
    return set(pd.read_csv(path)["key"]) if path.exists() else set()


def append(name, row):
    # Classification et régression n'ont pas les mêmes colonnes : le fichier est réécrit en entier à chaque ligne.
    path = OUT / f"{name}.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    print(name, row, flush=True)


def clean_mask(cat):
    """Spectre sans drapeau avec Teff et log g ASPCAP publiés : le critère « bon » sans la condition de SNR."""
    path = OUT / "clean.npy"
    if path.exists():
        return np.load(path)
    keys = ["APOGEE_ID", "TELESCOPE", "FIELD"]
    a = catalog.read_allstar().drop_duplicates(keys)
    a["clean"] = (a.STARFLAG == 0) & ~catalog.has_bit(a.ASPCAPFLAG, config.STAR_BAD_BIT) & np.isfinite(a.TEFF) & np.isfinite(a.LOGG)
    clean = cat[keys].merge(a[keys + ["clean"]], on=keys, how="left")["clean"].fillna(False).to_numpy(bool)
    np.save(path, clean)
    return clean


def scores(task, y, pred_fn, masks, cat):
    """Scores sur test_good, test_all et sur les spectres de test de SNR < 70."""
    low = masks["test_all"] & (cat.SNR < 70).to_numpy()
    out = {}
    for name, m in [("test_good", masks["test_good"]), ("test_all", masks["test_all"]), ("test_low", low)]:
        idx = np.flatnonzero(m)
        for metric, v in tasks.score(task, y[idx], pred_fn(idx)).items():
            out[f"{name}.{metric}"] = v
    return out


def train_cnn(X, cat, task, train_idx, epochs=30, lr=1e-3, track_epochs=False, seed=config.SEED):
    """Entraîne un CNN comme 06_train.py, garde la meilleure époque sur la validation, renvoie les scores de test."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    masks = tasks.split_masks(cat, task, "all")
    y = tasks.targets(cat, task)
    mu = sigma = None
    if task == "regress":
        mu, sigma = y[train_idx].mean(axis=0), y[train_idx].std(axis=0)
        y_fit = ((y - mu) / sigma).astype(np.float32)
        model, loss_fn = SpectrumCNN(len(config.REGRESS_TARGETS)), nn.MSELoss()
    else:
        y_fit = y
        model, loss_fn = SpectrumCNN(len(config.CLASSES)), nn.CrossEntropyLoss()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train = Batches(X, y_fit, train_idx, 128, shuffle=True)
    val = Batches(X, y_fit, np.flatnonzero(masks["val"]), 128, shuffle=False)

    def pred_fn(idx):
        out = predict(model, X, device, idx)
        return out.argmax(axis=1) if task == "class" else out * sigma + mu

    best, best_state, curve = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train, loss_fn, device, opt)
        vl = run_epoch(model, val, loss_fn, device)
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if track_epochs:
            model.eval()
            idx = np.flatnonzero(masks["test_good"])
            curve.append({"epoch": epoch, "train_loss": tr, "val_loss": vl,
                          **{f"test_good.{k}": v for k, v in tasks.score(task, y[idx], pred_fn(idx)).items()}})
            print(curve[-1], flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return {"n_train": len(train_idx), "best_val_loss": best, **scores(task, y, pred_fn, masks, cat)}, curve


def train_rf(X, cat, task, train_idx, n_components=50, n_trees=300):
    masks = tasks.split_masks(cat, task, "all")
    y = tasks.targets(cat, task)
    pca = spectra.fit_pca(X, train_idx, n_components)
    forest = RandomForestClassifier if task == "class" else RandomForestRegressor
    model = forest(n_estimators=n_trees, n_jobs=-1, random_state=config.SEED).fit(spectra.project(pca, X, train_idx), y[train_idx])
    return {"n_train": len(train_idx), **scores(task, y, lambda idx: model.predict(spectra.project(pca, X, idx)), masks, cat)}


def base_train(cat, task):
    """Jeu d'entraînement de référence : bons spectres du train, étiquetés pour la tâche."""
    m = tasks.split_masks(cat, task, "good")["train"]
    return np.flatnonzero(m)


def run_supervised(name, X, cat):
    done = done_keys(name)
    rng = np.random.default_rng(config.SEED)
    for value in GRID[name]:
        for task in TASKS if name != "lr" else ["class"]:
            key = f"{task}_{value}"
            if key in done:
                continue
            split_ok = tasks.split_masks(cat, task, "all")["train"]
            if name == "snr":
                idx = np.flatnonzero(split_ok & clean_mask(cat) & (cat.SNR >= value).to_numpy())
                res, _ = train_cnn(X, cat, task, idx)
            elif name == "badfrac":
                good = base_train(cat, task)
                bad = np.flatnonzero(split_ok & cat["quality"].eq("bad").to_numpy())
                extra = rng.permutation(bad)[: int(round(value * len(bad)))]
                res, _ = train_cnn(X, cat, task, np.sort(np.concatenate([good, extra])))
            elif name == "size":
                good = base_train(cat, task)
                idx = np.sort(np.random.default_rng(config.SEED).permutation(good)[: int(round(value * len(good)))])
                res, _ = train_cnn(X, cat, task, idx)
                append(name, {"key": f"rf_{key}", "model": "rf", "task": task, "value": value,
                              **train_rf(X, cat, task, idx)})
            elif name == "lr":
                res, _ = train_cnn(X, cat, task, base_train(cat, task), lr=value)
            elif name == "rf_trees":
                append(name, {"key": key, "model": "rf", "task": task, "value": value,
                              **train_rf(X, cat, task, base_train(cat, task), n_trees=value)})
                continue
            elif name == "rf_pca":
                append(name, {"key": key, "model": "rf", "task": task, "value": value,
                              **train_rf(X, cat, task, base_train(cat, task), n_components=value)})
                continue
            append(name, {"key": key, "model": "cnn", "task": task, "value": value, **res})


def run_epochs(X, cat):
    for task in TASKS:
        path = OUT / f"epochs_{task}.csv"
        if path.exists():
            continue
        _, curve = train_cnn(X, cat, task, base_train(cat, task), epochs=EPOCHS_MAX, track_epochs=True)
        pd.DataFrame(curve).to_csv(path, index=False)


def run_cluster(name, cat):
    good = cat["quality"].eq("good").to_numpy()
    Z_all = np.load(config.PCA_FILE)["Z"][good]
    labels = cat.loc[good, "class"].to_numpy()
    done = done_keys(name)

    def ext(pred):
        return {"ari": adjusted_rand_score(labels, pred), "nmi": normalized_mutual_info_score(labels, pred)}

    if name == "cluster_dims":
        for dims in GRID[name]:
            Z = StandardScaler().fit_transform(Z_all[:, :dims])
            for k in [3, 4]:
                for method in ["kmeans", "gmm"]:
                    key = f"{method}_{dims}_{k}"
                    if key in done:
                        continue
                    if method == "kmeans":
                        pred = KMeans(k, n_init=10, random_state=config.SEED).fit_predict(Z)
                    else:
                        pred = GaussianMixture(k, covariance_type="full", random_state=config.SEED).fit(Z).predict(Z)
                    append(name, {"key": key, "method": method, "dims": dims, "k": k, **ext(pred)})
    else:
        Z = StandardScaler().fit_transform(Z_all[:, :10])
        for k in GRID[name]:
            for seed in range(10):
                key = f"{k}_{seed}"
                if key in done:
                    continue
                gmm = GaussianMixture(k, covariance_type="full", random_state=seed).fit(Z)
                km = KMeans(k, n_init=1, random_state=seed).fit_predict(Z)
                g, kk = ext(gmm.predict(Z)), ext(km)
                append(name, {"key": key, "k": k, "seed": seed, "gmm_ari": g["ari"], "gmm_nmi": g["nmi"],
                              "gmm_bic": gmm.bic(Z), "kmeans_ari": kk["ari"], "kmeans_nmi": kk["nmi"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tests", nargs="+", choices=list(GRID) + ["epochs"])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    supervised = [t for t in args.tests if not t.startswith("cluster")]
    X, cat = spectra.load_dataset(with_spectra=bool(supervised))
    for t in args.tests:
        if t.startswith("cluster"):
            run_cluster(t, cat)
        elif t == "epochs":
            run_epochs(X, cat)
        else:
            run_supervised(t, X, cat)


if __name__ == "__main__":
    main()
