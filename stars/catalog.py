import numpy as np
import pandas as pd
from astropy.io import fits

from stars import config

ALLSTAR_COLUMNS = [
    "APOGEE_ID", "TELESCOPE", "FIELD", "GLON", "GLAT", "SNR", "STARFLAG",
    "ASPCAPFLAG", "EXTRATARG", "TEFF", "LOGG", "M_H", "ALPHA_M",
    "FE_H", "MG_FE", "GAIAEDR3_R_MED_GEO", "FPARAM",
]
# Position des paramètres dans le tableau FPARAM d'ASPCAP DR17.
FPARAM_INDEX = {"TEFF": 0, "LOGG": 1, "M_H": 3}
R0_KPC = 8.2  # distance Soleil-centre galactique


def _to_frame(table, columns):
    out = {}
    for name in columns:
        col = np.asarray(table[name])
        if col.dtype.kind == "S":
            col = np.char.strip(col.astype(str))
        elif col.dtype.byteorder == ">":
            col = col.astype(col.dtype.newbyteorder("="))
        if col.ndim == 2:
            # ASPCAP marque un ajustement raté par -9999.
            col = np.where(col < -1000, np.nan, col)
            out.update({f"{name}_{key}": col[:, i] for key, i in FPARAM_INDEX.items()})
        else:
            out[name] = col
    return pd.DataFrame(out)


def read_allstar(path=config.ALLSTAR):
    with fits.open(path, memmap=True) as hdul:
        return _to_frame(hdul[1].data, ALLSTAR_COLUMNS)


def read_rc_ids(path=config.RC_VAC):
    with fits.open(path, memmap=True) as hdul:
        return set(np.char.strip(np.asarray(hdul[1].data["APOGEE_ID"]).astype(str)))


def read_occam_members(path=config.OCCAM_VAC, min_prob=0.5):
    """APOGEE_ID -> nom de l'amas, pour les étoiles membres probables d'un amas ouvert.

    Une étoile est membre si ses probabilités RV, [Fe/H] et mouvement propre dépassent toutes min_prob.
    """
    with fits.open(path, memmap=True) as hdul:
        d = _to_frame(hdul[1].data, ["APOGEE_ID", "CLUSTER", "RV_PROB", "FEH_PROB", "PM_PROB"])
    member = (d.RV_PROB > min_prob) & (d.FEH_PROB > min_prob) & (d.PM_PROB > min_prob)
    d = d[member].drop_duplicates("APOGEE_ID")
    return dict(zip(d.APOGEE_ID, d.CLUSTER))


def has_bit(values, bit):
    return (np.asarray(values, dtype=np.int64) >> bit) & 1 == 1


def usable(df):
    """Étoiles à télécharger : télescopes de 2,5 m, sans doublons ni standards telluriques."""
    return (
        df.TELESCOPE.isin(["apo25m", "lco25m"])
        & ~has_bit(df.EXTRATARG, config.DUPLICATE_BIT)
        & ~has_bit(df.EXTRATARG, config.TELLURIC_BIT)
        & (df.SNR >= config.MIN_SNR)
    )


def is_good(df):
    return (
        (df.SNR >= config.GOOD_SNR)
        & (df.STARFLAG == 0)
        & ~has_bit(df.ASPCAPFLAG, config.STAR_BAD_BIT)
        & np.isfinite(df.TEFF)
        & np.isfinite(df.LOGG)
    )


def fill_labels(df):
    """Complète Teff, log g et [M/H] avec FPARAM quand ASPCAP ne publie pas de valeur.

    FPARAM est l'ajustement brut, publié même pour les étoiles STAR_BAD : il donne une étiquette approximative aux mauvais spectres.
    La colonne label_source garde l'origine : aspcap, fparam ou vide.
    """
    df = df.copy()
    source = np.where(np.isfinite(df.TEFF) & np.isfinite(df.LOGG), "aspcap", "")
    for key in FPARAM_INDEX:
        missing = ~np.isfinite(df[key])
        df.loc[missing, key] = df.loc[missing, f"FPARAM_{key}"]
    filled = (source == "") & np.isfinite(df.TEFF) & np.isfinite(df.LOGG)
    df["label_source"] = np.where(filled, "fparam", source)
    return df.drop(columns=[f"FPARAM_{key}" for key in FPARAM_INDEX])


def galactocentric_radius(df):
    """Distance au centre galactique (kpc), à partir de la distance Gaia EDR3 et de la position (l, b)."""
    d = df.GAIAEDR3_R_MED_GEO.to_numpy() / 1000
    l, b = np.radians(df.GLON.to_numpy()), np.radians(df.GLAT.to_numpy())
    x = R0_KPC - d * np.cos(b) * np.cos(l)
    y = d * np.cos(b) * np.sin(l)
    z = d * np.sin(b)
    return np.sqrt(x**2 + y**2 + z**2)


def evolutionary_class(df, rc_ids):
    logg = df.LOGG.to_numpy()
    label = np.full(len(df), "", dtype=object)
    label[logg >= 4.0] = "dwarf"
    label[(logg >= 3.5) & (logg < 4.0)] = "subgiant"
    giant = logg < 3.5
    in_rc = df.APOGEE_ID.isin(rc_ids).to_numpy()
    label[giant & in_rc] = "rc"
    label[giant & ~in_rc] = "rgb"
    return label


def population(df):
    """Population galactique selon [Mg/Fe], [Fe/H] et la distance au centre galactique.

    Disque mince ou épais : ligne de Weinberg et al. (2019) dans le plan [Mg/Fe] contre [Fe/H].
    Halo : [Fe/H] < -1. Bulbe : moins de 3 kpc du centre galactique.
    """
    fe_h, mg_fe = df.FE_H.to_numpy(), df.MG_FE.to_numpy()
    r_gc = df.R_GC.to_numpy()

    label = np.full(len(df), "", dtype=object)
    finite = np.isfinite(fe_h) & np.isfinite(mg_fe)
    high_alpha = mg_fe > 0.12 - 0.13 * np.minimum(fe_h, 0)
    label[finite & ~high_alpha] = "thin_disk"
    label[finite & high_alpha] = "thick_disk"
    label[finite & (fe_h < -1.0)] = "halo"
    label[finite & (fe_h >= -1.0) & (r_gc < 3.0)] = "bulge"
    return label


def assign_split(n, rng, fractions=(0.7, 0.15, 0.15)):
    split = np.empty(n, dtype=object)
    order = rng.permutation(n)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    split[order[:n_train]] = "train"
    split[order[n_train:n_train + n_val]] = "val"
    split[order[n_train + n_val:]] = "test"
    return split


def load_catalog():
    text = ["APOGEE_ID", "TELESCOPE", "FIELD", "quality", "class", "population", "occam_cluster", "split", "label_source"]
    df = pd.read_csv(config.CATALOG, dtype=dict.fromkeys(text, str))
    df[text] = df[text].fillna("")
    return df
