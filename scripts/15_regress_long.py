"""CNN de régression (Teff, log g, [M/H]) sur 100 époques, avec les meilleurs réglages de 12_best.py.

Le test de sensibilité sur 50 époques montrait une perte de validation encore en baisse. Ce script prolonge l'entraînement
à 100 époques, avec 3 graines, et note à chaque époque les pertes et l'erreur sur test_good.
Résultats dans STARS_DATA/regress_long/ (ou --out) : curve_<graine>.csv (une ligne par époque) et final.csv (une ligne par
graine). Une relance saute les graines déjà terminées. Les spectres sont lus en mémoire au départ : avec --out sur un disque
interne, une déconnexion du disque de données pendant l'entraînement n'arrête pas le run.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from stars import config, plots, spectra


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sens = load("sens", "11_sensitivity.py")
best = load("best", "12_best.py")

OUT = config.DATA / "regress_long"
FIGURE = config.ROOT / "report" / "figures" / "regress_long.png"


def plot(out=OUT):
    """Pertes et erreurs sur test_good, une couleur par graine. Trait fin : chaque époque. Trait épais : le modèle gardé si
    l'entraînement s'arrêtait à cette époque (meilleure perte de validation jusque-là). Trait vertical : 30 époques."""
    curves = {int(f.stem.split("_")[1]): pd.read_csv(f) for f in sorted(out.glob("curve_*.csv"))}
    panels = [("val_loss", "perte de validation (cibles standardisées)"), ("test_good.mae_TEFF", "erreur Teff (K)"),
              ("test_good.mae_LOGG", "erreur log g (dex)"), ("test_good.mae_M_H", "erreur [M/H] (dex)")]
    fig, axes = plots.plt.subplots(1, 4, figsize=(18, 4))
    for ax, (col, label) in zip(axes, panels):
        for (seed, c), color in zip(curves.items(), plots.plt.rcParams["axes.prop_cycle"].by_key()["color"]):
            kept = c.val_loss.expanding().apply(lambda v: v.argmin()).astype(int)
            ax.plot(c.epoch, c[col], lw=0.6, alpha=0.35, color=color)
            ax.step(c.epoch, c[col].to_numpy()[kept], where="post", lw=2, color=color, label=f"graine {seed}")
        ax.axvline(30, color="grey", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("époque")
        ax.set_title(label)
    axes[0].legend(fontsize=8)
    plots._save(fig, FIGURE)
    print(f"Figure dans {FIGURE}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--snr", type=float, default=50, help="seuil de SNR du jeu d'entraînement (choisi par 12_best.py)")
    p.add_argument("--badfrac", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=OUT, help="dossier des résultats")
    p.add_argument("--plot-only", action="store_true", help="refaire la figure à partir des courbes déjà écrites")
    args = p.parse_args()
    if args.plot_only:
        return plot(args.out)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    X, cat = spectra.load_dataset()
    idx = best.train_set(cat, "regress", args.snr, args.badfrac)
    path = out / "final.csv"
    done = set(pd.read_csv(path)["seed"]) if path.exists() else set()

    for seed in args.seeds:
        if seed in done:
            continue
        res, curve = sens.train_cnn(X, cat, "regress", idx, epochs=args.epochs, track_epochs=True, seed=seed)
        curve = pd.DataFrame(curve)
        curve.to_csv(out / f"curve_{seed}.csv", index=False)
        row = {"seed": seed, "epochs": args.epochs, "snr": args.snr, "badfrac": args.badfrac,
               "best_epoch": int(curve.val_loss.idxmin()) + 1, **res}
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
        print(row, flush=True)
    plot(out)
    print("=== REGRESS LONG DONE")


if __name__ == "__main__":
    main()
