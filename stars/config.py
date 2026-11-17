import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# STARS_DATA doit pointer vers un disque avec ~60 Go libres. Tous les scripts écrivent là.
DATA = Path(os.environ.get("STARS_DATA", ROOT / "data"))
RUNS = DATA / "runs"
FIGURES = DATA / "figures"

ALLSTAR = DATA / "allStar-dr17-synspec_rev1.fits"
RC_VAC = DATA / "apogee-rc-DR17.fits"
OCCAM_VAC = DATA / "occam_member-DR17.fits"
CATALOG = DATA / "catalog.csv"
FLUX = DATA / "flux.npy"
ERR = DATA / "err.npy"
STATUS = DATA / "status.npy"
FEATURES = DATA / "features.npz"
FEATURES_X = DATA / "features_X.npy"
PCA_FILE = DATA / "pca.npz"

SAS = "https://data.sdss.org/sas/dr17/apogee"
ASPCAP_DIR = f"{SAS}/spectro/aspcap/dr17/synspec_rev1"
URLS = {
    ALLSTAR: f"{ASPCAP_DIR}/allStar-dr17-synspec_rev1.fits",
    RC_VAC: f"{SAS}/vac/apogee-rc/cat/apogee-rc-DR17.fits",
    OCCAM_VAC: f"{SAS}/vac/apogee-occam/occam_member-DR17.fits",
}

N_PIX = 8575
SEED = 42

GOOD_SNR = 70
MIN_SNR = 5
STAR_BAD_BIT = 23  # ASPCAPFLAG
TELLURIC_BIT = 2  # EXTRATARG
DUPLICATE_BIT = 4  # EXTRATARG

CLASSES = ["dwarf", "subgiant", "rgb", "rc"]
POPULATIONS = ["thin_disk", "thick_disk", "halo", "bulge"]
REGRESS_TARGETS = ["TEFF", "LOGG", "M_H"]

# Codes d'état des spectres dans status.npy
TODO, OK, MISSING = 0, 1, 2
