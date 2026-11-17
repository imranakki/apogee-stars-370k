"""Choisit l'échantillon d'étoiles, attribue qualité, étiquettes et split, écrit catalog.csv."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from stars import catalog, config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-good", type=int, default=246200, help="nombre d'étoiles de bonne qualité")
    p.add_argument("--n-bad", type=int, default=123100, help="nombre d'étoiles de mauvaise qualité")
    p.add_argument("--no-occam", action="store_true", help="ne pas forcer l'inclusion des membres d'amas ouverts")
    p.add_argument("--force", action="store_true", help="écraser un catalog.csv existant")
    args = p.parse_args()

    # Les spectres sont rangés dans l'ordre de catalog.csv : le réécrire invalide flux.npy.
    if config.CATALOG.exists() and not args.force:
        sys.exit(f"{config.CATALOG} existe déjà. Ajoutez --force, puis supprimez flux.npy, err.npy et status.npy.")

    rng = np.random.default_rng(config.SEED)
    df = catalog.read_allstar()
    df = df[catalog.usable(df)].drop_duplicates("APOGEE_ID").reset_index(drop=True)
    # La qualité se décide avant de compléter les étiquettes avec FPARAM.
    df["quality"] = np.where(catalog.is_good(df), "good", "bad")
    df = catalog.fill_labels(df)
    df["R_GC"] = catalog.galactocentric_radius(df)
    df["class"] = catalog.evolutionary_class(df, catalog.read_rc_ids())
    df["population"] = catalog.population(df)
    df["occam_cluster"] = df.APOGEE_ID.map(catalog.read_occam_members()).fillna("")

    forced = df[df.occam_cluster != ""] if not args.no_occam else df.iloc[:0]
    rest = df.drop(forced.index)
    picks = [forced]
    for quality, n in [("good", args.n_good), ("bad", args.n_bad)]:
        pool = rest[rest.quality == quality]
        picks.append(pool.sample(n=min(n, len(pool)), random_state=config.SEED))
    sample = pd.concat(picks).reset_index(drop=True)
    sample["split"] = catalog.assign_split(len(sample), rng)

    sample.drop(columns=["STARFLAG", "ASPCAPFLAG", "EXTRATARG"]).to_csv(config.CATALOG, index=False)

    print(f"{len(sample)} étoiles écrites dans {config.CATALOG}")
    print(pd.crosstab(sample["class"].replace("", "?"), sample.quality))
    print(f"Membres OCCAM : {(sample.occam_cluster != '').sum()}")
    print(pd.crosstab(sample.label_source.replace("", "?"), sample.quality))
    print(pd.crosstab(sample.population.replace("", "?"), sample.quality))


if __name__ == "__main__":
    main()
