"""Sélection des lignes et cibles communes à 05_baseline et 06_train."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from stars import config


def labelled(cat, task):
    if task == "none":
        # L'autoencodeur n'utilise pas d'étiquettes : toutes les étoiles du split servent.
        return np.ones(len(cat), dtype=bool)
    if task == "class":
        return cat["class"].isin(config.CLASSES).to_numpy()
    return np.isfinite(cat[config.REGRESS_TARGETS].to_numpy(dtype=float)).all(axis=1)


def targets(cat, task):
    if task == "class":
        # -1 pour les étoiles sans classe : elles sont exclues par les masques.
        return cat["class"].map({c: i for i, c in enumerate(config.CLASSES)}).fillna(-1).to_numpy(dtype=np.int64)
    return cat[config.REGRESS_TARGETS].to_numpy(dtype=np.float32)


def split_masks(cat, task, train_on):
    """Masques booléens pour train, val et les deux jeux de test.

    La validation utilise toujours les bons spectres : chaque run choisit sa meilleure époque de la même façon.
    """
    ok = labelled(cat, task)
    good = cat["quality"].eq("good").to_numpy()
    split = cat["split"].to_numpy()
    train = ok & (split == "train")
    if train_on == "good":
        train &= good
    return {
        "train": train,
        "val": ok & good & (split == "val"),
        "test_good": ok & good & (split == "test"),
        "test_all": ok & (split == "test"),
    }


def score(task, y_true, y_pred):
    if task == "class":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "n": int(len(y_true)),
        }
    out = {f"mae_{name}": float(np.mean(np.abs(y_true[:, i] - y_pred[:, i]))) for i, name in enumerate(config.REGRESS_TARGETS)}
    out["n"] = int(len(y_true))
    return out
