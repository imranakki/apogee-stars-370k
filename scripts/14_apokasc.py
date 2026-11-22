"""Red clump contre RGB avec les étiquettes d'astérosismologie d'APOKASC-3 (Pinsonneault et al. 2025).

À lancer après 13_rc_rgb.py. Lit STARS_DATA/apokasc3/table4.tsv (VizieR J/ApJS/276/69), joint sur l'identifiant 2MASS.
Seuls les états évolutifs mesurés par l'astérosismologie (EvolSrc = Seis) servent : les états spectroscopiques
d'APOKASC-3 viennent eux-mêmes des spectres APOGEE.

Trois questions :
  1. Le catalogue red clump est-il juste ? Tableau croisé des deux étiquettes.
  2. Les modèles entraînés sur le catalogue, jugés contre l'astérosismologie, gardent-ils leur précision de 0,57 ?
  3. Un modèle entraîné sur les étiquettes d'astérosismologie fait-il mieux ?
Le test reste celui du projet : bons spectres du split test, avec une étiquette d'astérosismologie.
Résultats dans STARS_DATA/apokasc3/results.csv et crosstab.csv.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import nn

from stars import config, spectra
from stars.models import SpectrumCNN
from stars.training import Batches, predict, run_epoch

_spec = importlib.util.spec_from_file_location("rcrgb", Path(__file__).with_name("13_rc_rgb.py"))
rcrgb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcrgb)

DIR = config.DATA / "apokasc3"
PARAMS = rcrgb.PARAMS + rcrgb.CN


def download():
    """Table 4 d'APOKASC-3 depuis VizieR. STARS_VIZIER choisit un miroir si le serveur principal ne répond pas."""
    import os
    import urllib.request
    DIR.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("STARS_VIZIER", "https://vizier.cfa.harvard.edu")
    url = f"{host}/viz-bin/asu-tsv?-source=J/ApJS/276/69/table4&-out.max=unlimited&-out.all"
    urllib.request.urlretrieve(url, DIR / "table4.tsv")


def load_seismic(cat):
    """1 pour red clump, 0 pour RGB, -1 sans état d'astérosismologie."""
    if not (DIR / "table4.tsv").exists():
        download()
    t = pd.read_csv(DIR / "table4.tsv", sep="\t", comment="#", dtype=str).iloc[2:]
    t.columns = [c.strip() for c in t.columns]
    t = t.apply(lambda s: s.str.strip())
    t = t[(t.EvolSrc == "Seis") & t.EvolSt.isin(["RC", "RGB"])]
    t["APOGEE_ID"] = "2M" + t["2MASS"].str.lstrip("J")
    state = cat[["APOGEE_ID"]].merge(t[["APOGEE_ID", "EvolSt"]].drop_duplicates("APOGEE_ID"), on="APOGEE_ID", how="left")
    return state.EvolSt.map({"RC": 1, "RGB": 0}).fillna(-1).to_numpy(np.int64)


def masks_for(cat, has_label, good_only):
    split = cat["split"].to_numpy()
    good = cat["quality"].eq("good").to_numpy()
    base = has_label & (good if good_only else True)
    return {s: base & (split == s) for s in ["train", "val", "test"]}


def metrics(y, p, t):
    pred = p >= t
    return {"threshold": t, "rc_recall": recall_score(y, pred), "rc_precision": precision_score(y, pred),
            "rc_f1": f1_score(y, pred), "auc": roc_auc_score(y, p), "n_test": len(y), "n_test_rc": int(y.sum())}


def forest(features, y_fit, fit, val, y_val):
    ok = np.isfinite(features).all(axis=1)
    fit, val = fit[ok[fit]], val[ok[val]]
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1,
                                   random_state=config.SEED).fit(features[fit], y_fit[fit])
    t = rcrgb.best_threshold(y_val[val], model.predict_proba(features[val])[:, 1])
    return lambda idx: model.predict_proba(features[idx])[:, 1], t, len(fit)


def cnn(X, y_fit, fit, val, seed, init=None, epochs=30, lr=1e-3):
    """CNN binaire, perte pondérée. init : poids de départ (affinage). Garde la meilleure époque de validation."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    counts = np.bincount(y_fit[fit], minlength=2)
    weight = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    model = SpectrumCNN(2)
    if init is not None:
        model.load_state_dict(init)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train = Batches(X, y_fit, fit, 128, shuffle=True)
    valb = Batches(X, y_fit, val, 128, shuffle=False)
    best, state = float("inf"), None
    for epoch in range(1, epochs + 1):
        run_epoch(model, train, loss_fn, device, opt)
        vl = run_epoch(model, valb, loss_fn, device)
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()

    def proba(idx):
        return torch.softmax(torch.from_numpy(predict(model, X, device, idx)), dim=1)[:, 1].numpy()

    return proba, {k: v.cpu() for k, v in state.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    path = DIR / "results.csv"
    done = set(pd.read_csv(path)["key"]) if path.exists() else set()

    def save(row):
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
        print(row, flush=True)

    X, cat = spectra.load_dataset()
    cat = rcrgb.add_cn(cat)
    seis = load_seismic(cat)
    has = seis >= 0
    giants = rcrgb.giant_masks(cat)
    cat_y = cat["class"].eq("rc").to_numpy(np.int64)

    # 1. Étiquettes du catalogue contre astérosismologie, tous spectres confondus.
    cross = pd.crosstab(cat.loc[has, "class"], np.where(seis[has] == 1, "seis_RC", "seis_RGB"))
    cross.to_csv(DIR / "crosstab.csv")
    print(cross, flush=True)

    # Test commun : bons spectres du split test, étiquette d'astérosismologie, géante selon notre catalogue.
    giant_any = cat["class"].isin(["rgb", "rc"]).to_numpy()
    test = np.flatnonzero(has & giant_any & cat["quality"].eq("good").to_numpy() & (cat["split"] == "test").to_numpy())
    y_test = seis[test]
    # Le catalogue lui-même, vu comme un classifieur.
    if "catalog_labels" not in done:
        save({"key": "catalog_labels", "train_labels": "-", "model": "catalogue red clump", "seed": "-",
              "n_train": 0, **metrics(y_test, cat_y[test].astype(float), 0.5)})

    feats = cat[PARAMS].to_numpy(float)
    g_tr, g_va = np.flatnonzero(giants["train"]), np.flatnonzero(giants["val"])
    s_m = masks_for(cat, has & giant_any, good_only=False)
    s_tr, s_va = np.flatnonzero(s_m["train"]), np.flatnonzero(s_m["val"])

    # 2. et 3. Forêts sur paramètres ASPCAP + [C/Fe], [N/Fe].
    # Seuil toujours réglé sur la validation d'astérosismologie, pour les deux sources d'étiquettes.
    for key, y_fit, fit in [("rf_catalog", cat_y, g_tr), ("rf_seismic", seis, s_tr)]:
        if key not in done:
            proba, t, n = forest(feats, y_fit, fit, s_va, seis)
            save({"key": key, "train_labels": key.split("_")[1], "model": "forêt, paramètres + C, N", "seed": config.SEED,
                  "n_train": n, **metrics(y_test, proba(test), t)})

    # CNN : catalogue, astérosismologie seule, catalogue puis affinage sur l'astérosismologie.
    for seed in args.seeds:
        keys = [f"cnn_catalog_{seed}", f"cnn_seismic_{seed}", f"cnn_finetune_{seed}"]
        if all(k in done for k in keys):
            continue
        proba, state = cnn(X, cat_y, g_tr, g_va, seed)
        # Seuil réglé sur la validation d'astérosismologie : la question est la justesse contre la vraie étiquette.
        t = rcrgb.best_threshold(seis[s_va], proba(s_va))
        if keys[0] not in done:
            save({"key": keys[0], "train_labels": "catalog", "model": "CNN", "seed": seed, "n_train": len(g_tr),
                  **metrics(y_test, proba(test), t)})
        if keys[1] not in done:
            p2, _ = cnn(X, seis, s_tr, s_va, seed)
            save({"key": keys[1], "train_labels": "seismic", "model": "CNN", "seed": seed, "n_train": len(s_tr),
                  **metrics(y_test, p2(test), rcrgb.best_threshold(seis[s_va], p2(s_va)))})
        if keys[2] not in done:
            p3, _ = cnn(X, seis, s_tr, s_va, seed, init=state, lr=3e-4)
            save({"key": keys[2], "train_labels": "catalog+seismic", "model": "CNN affiné", "seed": seed,
                  "n_train": len(s_tr), **metrics(y_test, p3(test), rcrgb.best_threshold(seis[s_va], p3(s_va)))})
    print("=== APOKASC DONE")


if __name__ == "__main__":
    main()
