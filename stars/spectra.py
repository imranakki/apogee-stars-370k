import numpy as np
from sklearn.decomposition import PCA

from stars import catalog, config

MAX_ERR = 0.1
FLUX_CLIP = (0.0, 1.5)


def preprocess(flux, err, keep_pixels):
    """Retire les trous entre détecteurs, puis remplace les pixels douteux par le continuum (1,0)."""
    flux = flux[:, keep_pixels].astype(np.float32)
    err = err[:, keep_pixels]
    bad = ~np.isfinite(flux) | ~np.isfinite(err) | (flux <= 0) | (err > MAX_ERR)
    flux[bad] = 1.0
    np.clip(flux, *FLUX_CLIP, out=flux)
    return flux, bad.mean(axis=1)


def build_features(chunk=2000):
    """Prétraite une fois chaque spectre téléchargé. X va dans features_X.npy (float16), le reste dans features.npz."""
    status = np.load(config.STATUS)
    rows = np.flatnonzero(status == config.OK)
    flux = np.load(config.FLUX, mmap_mode="r")
    err = np.load(config.ERR, mmap_mode="r")

    # Les trous entre détecteurs valent zéro dans presque tous les spectres.
    sample = flux[rows[:: max(1, len(rows) // 2000)]]
    keep_pixels = np.flatnonzero((sample == 0).mean(axis=0) < 0.5)

    # float16 : 370 000 spectres tiennent en 5,6 Go de RAM. Le pas de 0,0005 près du continuum reste sous le bruit.
    X = np.lib.format.open_memmap(config.FEATURES_X, mode="w+", dtype=np.float16, shape=(len(rows), len(keep_pixels)))
    bad_frac = np.empty(len(rows), dtype=np.float32)
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        X[i:i + chunk], bad_frac[i:i + chunk] = preprocess(flux[r], err[r], keep_pixels)
    X.flush()
    del X
    np.savez(config.FEATURES, rows=rows, keep_pixels=keep_pixels, bad_frac=bad_frac)


def load_dataset(with_spectra=True):
    """Renvoie (X, cat) : spectres prétraités (float16) et lignes du catalogue correspondantes, dans le même ordre.

    with_spectra=False renvoie (None, cat) sans charger les 5,6 Go de spectres.
    """
    if not config.FEATURES.exists() or config.FEATURES.stat().st_mtime < config.STATUS.stat().st_mtime:
        build_features()
    data = np.load(config.FEATURES)
    cat = catalog.load_catalog().iloc[data["rows"]].reset_index(drop=True)
    cat["bad_pixel_frac"] = data["bad_frac"]
    return (np.load(config.FEATURES_X) if with_spectra else None), cat


def fit_pca(X, idx, n_components, n_fit=50000):
    """PCA ajustée sur au plus n_fit spectres tirés dans idx : la matrice complète ne tient pas en RAM en float32."""
    rng = np.random.default_rng(config.SEED)
    fit_idx = np.sort(rng.choice(idx, size=min(n_fit, len(idx)), replace=False))
    return PCA(n_components=n_components, random_state=config.SEED).fit(X[fit_idx].astype(np.float32))


def project(pca, X, idx=None, chunk=20000):
    """Projette X[idx] (tout X par défaut) sur la PCA, par morceaux convertis en float32."""
    idx = np.arange(len(X)) if idx is None else idx
    return np.concatenate([pca.transform(X[idx[i:i + chunk]].astype(np.float32)) for i in range(0, len(idx), chunk)])


def wavelengths(keep_pixels=None):
    """Longueur d'onde (Å, vide) de chaque pixel, selon l'en-tête aspcapStar (grille log-linéaire)."""
    wave = 10 ** (4.179 + 6e-6 * np.arange(config.N_PIX))
    return wave if keep_pixels is None else wave[keep_pixels]
