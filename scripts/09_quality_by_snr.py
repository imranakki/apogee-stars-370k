"""Performance des CNN par tranche de SNR sur le test complet. Compare les modèles entraînés sur « good » et sur « all »."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from stars import config, plots, spectra, tasks
from stars.models import SpectrumCNN
from stars.training import predict

SNR_BINS = [5, 30, 70, 150, np.inf]


def evaluate(run, X, cat, device):
    cfg = json.loads((config.RUNS / run / "config.json").read_text())
    task = cfg["task"]
    masks = tasks.split_masks(cat, task, cfg["train_on"])
    y = tasks.targets(cat, task)
    n_out = len(config.CLASSES) if task == "class" else len(config.REGRESS_TARGETS)
    model = SpectrumCNN(n_out)
    model.load_state_dict(torch.load(config.RUNS / run / "best.pt", map_location="cpu", weights_only=False)["model"])
    model.to(device).eval()

    test = masks["test_all"]
    out = predict(model, X, device, np.flatnonzero(test))
    if task == "class":
        pred = out.argmax(axis=1)
    else:
        # Même standardisation qu'à l'entraînement (statistiques du train).
        mu, sigma = y[masks["train"]].mean(axis=0), y[masks["train"]].std(axis=0)
        pred = out * sigma + mu

    snr = cat["SNR"].to_numpy()[test]
    rows = []
    for lo, hi in zip(SNR_BINS, SNR_BINS[1:]):
        b = (snr >= lo) & (snr < hi)
        if b.sum() >= 20:
            rows.append({"run": run, "snr_bin": f"{lo:g}-{hi:g}", "snr_lo": lo, **tasks.score(task, y[test][b], pred[b])})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=["cnn_class_good", "cnn_class_all", "cnn_regress_good", "cnn_regress_all"])
    args = p.parse_args()

    X, cat = spectra.load_dataset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = pd.DataFrame([row for run in args.runs for row in evaluate(run, X, cat, device)])
    table.to_csv(config.DATA / "quality_by_snr.csv", index=False)
    print(table.round(3).to_string(index=False))

    out = config.FIGURES / "quality"
    for metric in ["macro_f1", "mae_LOGG", "mae_TEFF"]:
        if metric not in table:
            continue
        sub = table.dropna(subset=[metric])
        runs = list(dict.fromkeys(sub.run))
        bins = list(dict.fromkeys(sub.snr_bin))
        ys = [sub[sub.run == r].set_index("snr_bin").reindex(bins)[metric].to_numpy() for r in runs]
        plots.save_line_plot(np.arange(len(bins)), ys, runs, out / f"{metric}_by_snr.png",
                             "SNR", metric, f"{metric} par tranche de SNR (test complet)", xticklabels=bins)
    print(f"Tableau : {config.DATA / 'quality_by_snr.csv'}, figures : {out}")


if __name__ == "__main__":
    main()
