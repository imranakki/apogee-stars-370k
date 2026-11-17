"""Référence supervisée : PCA puis forêt aléatoire. Sert de point de comparaison pour le CNN."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from stars import config, spectra, tasks
from stars.training import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["class", "regress"], default="class")
    p.add_argument("--train-on", choices=["good", "all"], default="good")
    p.add_argument("--n-components", type=int, default=50)
    p.add_argument("--n-trees", type=int, default=300)
    args = p.parse_args()

    X, cat = spectra.load_dataset()
    masks = tasks.split_masks(cat, args.task, args.train_on)
    y = tasks.targets(cat, args.task)

    train = np.flatnonzero(masks["train"])
    # PCA ajustée sur le train seulement, pour ne rien apprendre du test.
    pca = spectra.fit_pca(X, train, args.n_components)
    forest = RandomForestClassifier if args.task == "class" else RandomForestRegressor
    model = forest(n_estimators=args.n_trees, n_jobs=-1, random_state=config.SEED)
    model.fit(spectra.project(pca, X, train), y[train])

    results = {"model": "random_forest", **vars(args), "n_train": int(masks["train"].sum())}
    for name in ["test_good", "test_all"]:
        m = np.flatnonzero(masks[name])
        results[name] = tasks.score(args.task, y[m], model.predict(spectra.project(pca, X, m)))
        print(name, results[name])

    run_dir = config.RUNS / f"rf_{args.task}_{args.train_on}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "metrics.json", results)
    print(f"Résultats dans {run_dir}")


if __name__ == "__main__":
    main()
