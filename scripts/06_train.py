"""Entraîne un CNN 1-D (classification ou régression) ou un autoencodeur. Relancer la même commande reprend l'entraînement."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from stars import config, spectra, tasks
from stars.models import Autoencoder, SpectrumCNN
from stars.training import Batches, Checkpointer, predict, run_epoch, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cnn", "autoencoder"], default="cnn")
    p.add_argument("--task", choices=["class", "regress"], default="class", help="ignoré pour l'autoencodeur")
    p.add_argument("--train-on", choices=["good", "all"], default="good")
    p.add_argument("--epochs", type=int, default=30, help="nombre total d'époques, reprise comprise")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--run-name", help="nom du dossier dans runs/ (par défaut : déduit des options)")
    p.add_argument("--fresh", action="store_true", help="ignorer un checkpoint existant et repartir de zéro")
    return p.parse_args()


def main():
    args = parse_args()
    autoencoder = args.model == "autoencoder"
    if autoencoder:
        args.task = "none"
    name = args.run_name or f"{args.model}_{args.task}_{args.train_on}"
    run_dir = config.RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, cat = spectra.load_dataset()

    masks = tasks.split_masks(cat, args.task, args.train_on)
    if autoencoder:
        y, model, loss_fn = None, Autoencoder(X.shape[1], args.latent_dim), nn.MSELoss()
    else:
        y = tasks.targets(cat, args.task)
        if args.task == "class":
            model, loss_fn = SpectrumCNN(len(config.CLASSES)), nn.CrossEntropyLoss()
        else:
            # Cibles standardisées avec les statistiques du train, puis remises en unités physiques à l'évaluation.
            mu, sigma = y[masks["train"]].mean(axis=0), y[masks["train"]].std(axis=0)
            y = ((y - mu) / sigma).astype(np.float32)
            model, loss_fn = SpectrumCNN(len(config.REGRESS_TARGETS)), nn.MSELoss()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt = Checkpointer(run_dir, model, optimizer)
    if not args.fresh and ckpt.resume():
        print(f"Reprise de {run_dir.name} à l'époque {ckpt.epoch}")
    write_json(run_dir / "config.json", vars(args))

    train_data = Batches(X, y, np.flatnonzero(masks["train"]), args.batch_size, shuffle=True)
    val_data = Batches(X, y, np.flatnonzero(masks["val"]), args.batch_size, shuffle=False)
    print(f"{device} | train {masks['train'].sum()} | val {masks['val'].sum()}")

    while ckpt.epoch < args.epochs:
        train_loss = run_epoch(model, train_data, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_data, loss_fn, device)
        ckpt.save(train_loss, val_loss)
        print(f"époque {ckpt.epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}", flush=True)

    ckpt.load_best()
    model.eval()
    results = {**vars(args), "n_train": int(masks["train"].sum()), "best_val_loss": ckpt.best_val}

    if autoencoder:
        latent = predict(model.encode, X, device)
        np.save(run_dir / "latent.npy", latent)
        print(f"Vecteurs latents ({latent.shape}) dans {run_dir / 'latent.npy'}")
    else:
        for split_name in ["test_good", "test_all"]:
            m = masks[split_name]
            out = predict(model, X, device, np.flatnonzero(m))
            if args.task == "class":
                pred, true = out.argmax(axis=1), y[m]
            else:
                pred, true = out * sigma + mu, y[m] * sigma + mu
            results[split_name] = tasks.score(args.task, true, pred)
            print(split_name, results[split_name])

    write_json(run_dir / "metrics.json", results)


if __name__ == "__main__":
    main()
