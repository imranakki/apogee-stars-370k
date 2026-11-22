"""Classifieur binaire red clump contre RGB, sur les géantes à bon spectre.

Quatre modèles comparés sur le même découpage train / val / test :
  rf_params    forêt sur Teff, log g, [M/H] d'ASPCAP : ce que les étiquettes savent déjà.
  rf_params_cn forêt sur les mêmes paramètres plus [C/Fe] et [N/Fe] : le mélange après le premier dredge-up.
  rf_pca       forêt sur 10 composantes PCA du spectre.
  cnn          CNN 1-D sur le spectre complet, perte pondérée par l'inverse de la fréquence de classe.

Le seuil de décision maximise le F1 du red clump sur la validation, puis le test sert une seule fois.
Résultats dans STARS_DATA/rc_rgb/results.csv, une ligne par modèle et par graine. Une relance saute les lignes écrites.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import nn

from stars import catalog, config, spectra
from stars.models import SpectrumCNN
from stars.training import Batches, predict, run_epoch

OUT = config.DATA / "rc_rgb"
PARAMS = ["TEFF", "LOGG", "M_H"]
CN = ["C_FE", "N_FE"]
# Fenêtre où red clump et RGB se recouvrent dans le diagramme de Kiel.
WINDOW = (2.2, 2.8)


def giant_masks(cat):
    giant = cat["class"].isin(["rgb", "rc"]).to_numpy() & cat["quality"].eq("good").to_numpy()
    split = cat["split"].to_numpy()
    return {s: giant & (split == s) for s in ["train", "val", "test"]}


def add_cn(cat):
    """Ajoute [C/Fe] et [N/Fe] d'allStar, joints sur l'identifiant, le télescope et le champ."""
    keys = ["APOGEE_ID", "TELESCOPE", "FIELD"]
    with fits.open(config.ALLSTAR, memmap=True) as hdul:
        a = catalog._to_frame(hdul[1].data, keys + CN).drop_duplicates(keys)
    return cat.merge(a, on=keys, how="left")


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])


def scores(y, p, t, logg):
    """Scores au seuil t sur tout le test, puis dans la fenêtre de recouvrement en log g."""
    out = {"threshold": t}
    win = (logg >= WINDOW[0]) & (logg < WINDOW[1])
    for name, m in [("test", np.ones(len(y), bool)), ("window", win)]:
        pred = p[m] >= t
        out.update({
            f"{name}.rc_recall": recall_score(y[m], pred),
            f"{name}.rc_precision": precision_score(y[m], pred),
            f"{name}.rc_f1": f1_score(y[m], pred),
            f"{name}.auc": roc_auc_score(y[m], p[m]),
            f"{name}.n": int(m.sum()),
            f"{name}.n_rc": int(y[m].sum()),
        })
    return out


def run_forest(features, y, masks, logg):
    ok = np.isfinite(features).all(axis=1)
    tr, va, te = (np.flatnonzero(masks[s] & ok) for s in ["train", "val", "test"])
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1,
                                   random_state=config.SEED).fit(features[tr], y[tr])
    t = best_threshold(y[va], model.predict_proba(features[va])[:, 1])
    return {"n_train": len(tr), **scores(y[te], model.predict_proba(features[te])[:, 1], t, logg[te])}


def run_cnn(X, y, masks, logg, seed, epochs=30, lr=1e-3):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = (np.flatnonzero(masks[s]) for s in ["train", "val", "test"])
    counts = np.bincount(y[tr], minlength=2)
    weight = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    model = SpectrumCNN(2).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train = Batches(X, y, tr, 128, shuffle=True)
    val = Batches(X, y, va, 128, shuffle=False)
    best, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        run_epoch(model, train, loss_fn, device, opt)
        vl = run_epoch(model, val, loss_fn, device)
        print(f"seed {seed} epoch {epoch} val {vl:.4f}", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    def proba(idx):
        return torch.softmax(torch.from_numpy(predict(model, X, device, idx)), dim=1)[:, 1].numpy()

    t = best_threshold(y[va], proba(va))
    return {"n_train": len(tr), "best_val_loss": best, **scores(y[te], proba(te), t, logg[te])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.csv"
    done = set(pd.read_csv(path)["key"]) if path.exists() else set()

    def save(row):
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
        print(row, flush=True)

    X, cat = spectra.load_dataset()
    cat = add_cn(cat)
    masks = giant_masks(cat)
    y = cat["class"].eq("rc").to_numpy(np.int64)
    logg = cat["LOGG"].to_numpy()

    for key, cols in [("rf_params", PARAMS), ("rf_params_cn", PARAMS + CN)]:
        if key not in done:
            save({"key": key, "model": key, "seed": config.SEED,
                  **run_forest(cat[cols].to_numpy(float), y, masks, logg)})
    if "rf_pca" not in done:
        giants = np.flatnonzero(masks["train"] | masks["val"] | masks["test"])
        pca = spectra.fit_pca(X, np.flatnonzero(masks["train"]), 10)
        Z = np.full((len(cat), 10), np.nan)
        Z[giants] = spectra.project(pca, X, giants)
        save({"key": "rf_pca", "model": "rf_pca", "seed": config.SEED, **run_forest(Z, y, masks, logg)})
    for seed in args.seeds:
        key = f"cnn_{seed}"
        if key not in done:
            save({"key": key, "model": "cnn", "seed": seed, **run_cnn(X, y, masks, logg, seed)})
    print("=== RC_RGB DONE")


if __name__ == "__main__":
    main()
