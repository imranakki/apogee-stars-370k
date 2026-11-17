import csv
import json

import numpy as np
import torch


class Checkpointer:
    """Écrit last.pt à chaque époque et best.pt à chaque nouvelle meilleure perte de validation. Un run interrompu reprend où il s'est arrêté."""

    def __init__(self, run_dir, model, optimizer):
        self.dir = run_dir
        self.last = run_dir / "last.pt"
        self.best = run_dir / "best.pt"
        self.model = model
        self.optimizer = optimizer
        self.epoch = 0
        self.best_val = float("inf")
        self.history = []

    def resume(self):
        if not self.last.exists():
            return False
        state = torch.load(self.last, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch = state["epoch"]
        self.best_val = state["best_val"]
        self.history = state["history"]
        return True

    def save(self, train_loss, val_loss):
        self.epoch += 1
        self.history.append({"epoch": self.epoch, "train_loss": train_loss, "val_loss": val_loss})
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "best_val": min(self.best_val, val_loss),
            "history": self.history,
        }
        if val_loss < self.best_val:
            self.best_val = val_loss
            torch.save(state, self.best)
        # Écriture dans un fichier temporaire d'abord : un plantage pendant la sauvegarde ne corrompt pas last.pt.
        tmp = self.last.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(self.last)
        with open(self.dir / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(self.history)

    def load_best(self):
        self.model.load_state_dict(torch.load(self.best, map_location="cpu", weights_only=False)["model"])


def depth(flux):
    """Profondeur des raies (1 - flux) en float32 : l'entrée de tous les réseaux."""
    return 1.0 - flux.astype(np.float32)


class Batches:
    """Lots tirés de X (float16) ligne par ligne : le sous-ensemble d'entraînement n'est jamais copié en entier."""

    def __init__(self, X, y, idx, batch_size, shuffle):
        self.X, self.y, self.idx = X, y, np.asarray(idx)
        self.batch_size, self.shuffle = batch_size, shuffle
        self.rng = np.random.default_rng(42)

    def __len__(self):
        return -(-len(self.idx) // self.batch_size)

    def __iter__(self):
        order = self.rng.permutation(self.idx) if self.shuffle else self.idx
        for i in range(0, len(order), self.batch_size):
            # Lignes triées : la lecture en mémoire est plus rapide, et y suit le même ordre.
            b = np.sort(order[i:i + self.batch_size])
            x = torch.from_numpy(depth(self.X[b]))
            yield (x,) if self.y is None else (x, torch.from_numpy(self.y[b]))


def run_epoch(model, data, loss_fn, device, optimizer=None):
    """Une passe sur les données. Entraîne si optimizer est fourni, sinon évalue. Renvoie la perte moyenne."""
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data:
            x = batch[0].to(device)
            target = batch[1].to(device) if len(batch) > 1 else x
            loss = loss_fn(model(x), target)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
    return total / count


@torch.no_grad()
def predict(fn, X, device, idx=None, batch_size=1024):
    """Applique fn (un modèle ou model.encode) à X[idx] (tout X par défaut) par lots, renvoie un tableau numpy."""
    idx = np.arange(len(X)) if idx is None else idx
    out = [fn(x.to(device)).cpu() for (x,) in Batches(X, None, idx, batch_size, shuffle=False)]
    return torch.cat(out).numpy()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))
