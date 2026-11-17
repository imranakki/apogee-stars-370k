"""Erreurs des CNN de régression sur le test complet, séparées par origine de l'étiquette (aspcap ou fparam) et par qualité."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from stars import config, spectra, tasks
from stars.models import SpectrumCNN
from stars.training import predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=["cnn_regress_good", "cnn_regress_all"])
    args = p.parse_args()

    X, cat = spectra.load_dataset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for run in args.runs:
        cfg = json.loads((config.RUNS / run / "config.json").read_text())
        masks = tasks.split_masks(cat, "regress", cfg["train_on"])
        y = tasks.targets(cat, "regress")
        mu, sigma = y[masks["train"]].mean(axis=0), y[masks["train"]].std(axis=0)
        model = SpectrumCNN(len(config.REGRESS_TARGETS))
        model.load_state_dict(torch.load(config.RUNS / run / "best.pt", map_location="cpu", weights_only=False)["model"])
        model.to(device).eval()
        test = masks["test_all"]
        pred = predict(model, X, device, np.flatnonzero(test)) * sigma + mu
        sub = cat[test]
        for source, quality in sub.groupby(["label_source", "quality"]).groups:
            m = (sub.label_source == source).to_numpy() & (sub.quality == quality).to_numpy()
            rows.append({"run": run, "label_source": source, "quality": quality,
                         **tasks.score("regress", y[test][m], pred[m])})
    table = pd.DataFrame(rows)
    table.to_csv(config.DATA / "label_source.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
