"""Choisit les meilleurs réglages sur la validation, puis les mesure une seule fois sur le test.

À lancer après 11_sensitivity.py. Chaque CNN s'entraîne avec 3 graines : le rapport donne la moyenne et l'écart type.
Résultats dans STARS_DATA/best/ : selection.json (réglages et raisons) et final.csv (une ligne par run).
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score

from stars import config, spectra, tasks

_spec = importlib.util.spec_from_file_location("sens", Path(__file__).with_name("11_sensitivity.py"))
sens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sens)

OUT = config.DATA / "best"
SEEDS = [42, 43, 44]
BASE = {"snr": 70, "badfrac": 0.0, "lr": 1e-3, "rf_pca": 50, "rf_trees": 300}


def pick_cnn(task):
    """Pour chaque test CNN, la valeur de plus faible perte de validation. La validation ne contient que des bons spectres."""
    choice, why = {}, {}
    for name in ["snr", "badfrac", "lr"]:
        df = pd.read_csv(sens.OUT / f"{name}.csv")
        df = df[(df.task == task) & (df.model == "cnn")]
        if df.empty:
            choice[name] = BASE[name]
            why[name] = "non testé pour cette tâche : valeur de référence"
            continue
        best = df.loc[df.best_val_loss.idxmin()]
        choice[name] = float(best.value)
        why[name] = f"perte de validation minimale ({best.best_val_loss:.4f}) parmi {sorted(df.value.tolist())}"
    return choice, why


def rf_val_score(X, cat, task, n_components, n_trees):
    """Score de validation d'une forêt : macro F1 pour les classes, erreur standardisée moyenne pour la régression."""
    masks = tasks.split_masks(cat, task, "good")
    y = tasks.targets(cat, task)
    train, val = np.flatnonzero(masks["train"]), np.flatnonzero(masks["val"])
    pca = spectra.fit_pca(X, train, n_components)
    forest = RandomForestClassifier if task == "class" else RandomForestRegressor
    model = forest(n_estimators=n_trees, n_jobs=-1, random_state=config.SEED).fit(spectra.project(pca, X, train), y[train])
    pred = model.predict(spectra.project(pca, X, val))
    if task == "class":
        return float(f1_score(y[val], pred, average="macro"))
    sigma = y[train].std(axis=0)
    return float(-np.mean(np.abs(pred - y[val]) / sigma))


def pick_rf(X, cat, task):
    path = OUT / f"rf_val_{task}.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    seen = {(r["n_components"], r["n_trees"]) for r in rows}
    for nc in [10, 20, 50]:
        for nt in [300, 500]:
            if (nc, nt) not in seen:
                rows.append({"n_components": nc, "n_trees": nt, "val_score": rf_val_score(X, cat, task, nc, nt)})
                pd.DataFrame(rows).to_csv(path, index=False)
                print(task, rows[-1], flush=True)
    df = pd.DataFrame(rows)
    best = df.loc[df.val_score.idxmax()]
    metric = "macro F1" if task == "class" else "erreur standardisée (signe inversé)"
    return ({"rf_pca": int(best.n_components), "rf_trees": int(best.n_trees)},
            f"meilleur score de validation ({metric} {best.val_score:.4f}) parmi 10, 20, 50 composantes et 300, 500 arbres")


def cluster_choice():
    """k par la silhouette, puis parmi les 10 graines du GMM la plus faible BIC. Aucune étiquette ne sert au choix."""
    df = pd.read_csv(sens.OUT / "cluster_k.csv")
    k3 = df[df.k == 3]
    best = k3.loc[k3.gmm_bic.idxmin()]
    return {"dims": 10, "k": 3, "seed": int(best.seed), "ari": float(best.gmm_ari), "nmi": float(best.gmm_nmi),
            "ari_median_10_seeds": float(k3.gmm_ari.median()),
            "why": "k = 3 : silhouette maximale. Graine : BIC minimale parmi 10, sans regarder les étiquettes."}


def train_set(cat, task, snr, badfrac):
    ok = tasks.split_masks(cat, task, "all")["train"]
    good = np.flatnonzero(ok & sens.clean_mask(cat) & (cat.SNR >= snr).to_numpy())
    if badfrac == 0:
        return good
    bad = np.flatnonzero(ok & cat["quality"].eq("bad").to_numpy() & ~np.isin(np.arange(len(cat)), good))
    extra = np.random.default_rng(config.SEED).permutation(bad)[: int(round(badfrac * len(bad)))]
    return np.sort(np.concatenate([good, extra]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, cat = spectra.load_dataset()
    selection = {"cluster": cluster_choice()}
    for task in sens.TASKS:
        cnn, why = pick_cnn(task)
        rf, rf_why = pick_rf(X, cat, task)
        selection[task] = {"cnn": cnn, "cnn_why": why, "rf": rf, "rf_why": rf_why}
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    print(json.dumps(selection, indent=2, ensure_ascii=False), flush=True)

    path = OUT / "final.csv"
    done = set(pd.read_csv(path)["key"]) if path.exists() else set()

    def save(row):
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
        print(row, flush=True)

    for task in sens.TASKS:
        sel = selection[task]
        for setting, rf in [("reference", {"rf_pca": BASE["rf_pca"], "rf_trees": BASE["rf_trees"]}), ("best", sel["rf"])]:
            key = f"rf_{task}_{setting}"
            if key not in done:
                res = sens.train_rf(X, cat, task, sens.base_train(cat, task), rf["rf_pca"], rf["rf_trees"])
                save({"key": key, "model": "rf", "task": task, "setting": setting, "seed": config.SEED, **rf, **res})
        configs = [("reference", {k: BASE[k] for k in ["snr", "badfrac", "lr"]}), ("best", sel["cnn"])]
        if configs[1][1] == configs[0][1]:
            configs = configs[:1]
        for setting, c in configs:
            idx = train_set(cat, task, c["snr"], c["badfrac"])
            for seed in SEEDS:
                key = f"cnn_{task}_{setting}_{seed}"
                if key in done:
                    continue
                res, _ = sens.train_cnn(X, cat, task, idx, lr=c["lr"], seed=seed)
                save({"key": key, "model": "cnn", "task": task, "setting": setting, "seed": seed, **c, **res})
    print("=== BEST DONE")


if __name__ == "__main__":
    main()
