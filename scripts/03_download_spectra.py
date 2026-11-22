"""Télécharge les spectres aspcapStar de catalog.csv dans flux.npy et err.npy. Relancer reprend le téléchargement.

--seed-from copie d'abord les spectres déjà présents dans un autre dossier de données.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from stars import catalog, config
from stars.download import fetch_spectrum


def open_array(path, shape, dtype):
    if path.exists():
        return np.load(path, mmap_mode="r+")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def seed_from(other, cat, flux, err, status, chunk=2000):
    """Copie les spectres OK d'un autre dossier de données vers les lignes TODO de ce catalogue."""
    key = ["APOGEE_ID", "TELESCOPE", "FIELD"]
    ocat = pd.read_csv(other / "catalog.csv", usecols=key, dtype=str)
    ok = np.load(other / "status.npy") == config.OK
    where = pd.Series(np.arange(len(ocat)), index=pd.MultiIndex.from_frame(ocat))[ok]
    src = where.reindex(pd.MultiIndex.from_frame(cat[key])).to_numpy()
    rows = np.flatnonzero(~np.isnan(src) & (status == config.TODO))
    src = src[rows].astype(np.int64)
    oflux = np.load(other / "flux.npy", mmap_mode="r")
    oerr = np.load(other / "err.npy", mmap_mode="r")
    # Lecture dans l'ordre du fichier source : le disque lit en continu au lieu de sauter.
    order = np.argsort(src)
    for i in range(0, len(order), chunk):
        o = order[i:i + chunk]
        flux[rows[o]] = oflux[src[o]]
        err[rows[o]] = oerr[src[o]]
    status[rows] = config.OK
    print(f"{len(rows)} spectres copiés depuis {other}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=16, help="téléchargements en parallèle")
    p.add_argument("--seed-from", type=Path, help="autre dossier de données avec des spectres déjà téléchargés")
    p.add_argument("--retry-missing", action="store_true", help="retenter les étoiles marquées absentes")
    args = p.parse_args()

    cat = catalog.load_catalog()
    n = len(cat)
    flux = open_array(config.FLUX, (n, config.N_PIX), np.float32)
    err = open_array(config.ERR, (n, config.N_PIX), np.float32)
    status = np.load(config.STATUS) if config.STATUS.exists() else np.zeros(n, dtype=np.int8)
    if len(status) != n:
        sys.exit("status.npy ne correspond pas à catalog.csv. Supprimez flux.npy, err.npy et status.npy.")

    if args.seed_from:
        seed_from(args.seed_from, cat, flux, err, status)
        flux.flush()
        err.flush()
        np.save(config.STATUS, status)

    todo = status == config.TODO
    if args.retry_missing:
        todo |= status == config.MISSING
    rows = np.flatnonzero(todo)
    print(f"{n - len(rows)}/{n} déjà traités, {len(rows)} à télécharger")

    def save():
        flux.flush()
        err.flush()
        np.save(config.STATUS, status)

    failures, start = 0, time.time()
    pool = ThreadPoolExecutor(args.workers)
    try:
        jobs = {pool.submit(fetch_spectrum, cat.TELESCOPE[r], cat.FIELD[r], cat.APOGEE_ID[r]): r for r in rows}
        for i, job in enumerate(as_completed(jobs), 1):
            r = jobs[job]
            try:
                result = job.result()
            except Exception as e:
                # Erreur réseau : la ligne reste TODO et sera retentée au prochain lancement.
                failures += 1
                print(f"  échec {cat.APOGEE_ID[r]} : {e}")
                continue
            if result is None:
                status[r] = config.MISSING
            else:
                flux[r], err[r] = result
                status[r] = config.OK
            if i % 500 == 0:
                save()
                rate = i / (time.time() - start)
                print(f"  {i}/{len(rows)}  {rate:.1f} spectres/s", flush=True)
    except KeyboardInterrupt:
        print("Interrompu. Relancez le script pour reprendre.")
    finally:
        # Sans cancel_futures, Ctrl-C attendrait la fin de toute la file.
        pool.shutdown(wait=False, cancel_futures=True)
        save()

    counts = np.bincount(status, minlength=3)
    print(f"OK {counts[config.OK]}, absents {counts[config.MISSING]}, restants {counts[config.TODO]} (échecs réseau : {failures})")


if __name__ == "__main__":
    main()
