import io
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
from astropy.io import fits

from stars import config

USER_AGENT = "apogee-stars-course-project"


def _open(url, start=0, timeout=60):
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)


def download_file(url, dest: Path):
    """Télécharge url vers dest. Un fichier partiel laissé par une interruption est complété, pas recommencé."""
    if dest.exists():
        return
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    with _open(url, start) as response:
        total = start + int(response.headers.get("Content-Length", 0))
        # Un serveur qui ignore Range répond 200 avec le fichier entier.
        mode = "ab" if response.status == 206 else "wb"
        with open(part, mode) as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    if part.stat().st_size != total:
        raise IOError(f"{dest.name} : {part.stat().st_size} octets sur {total}, relancez pour continuer")
    part.rename(dest)


def spectrum_url(telescope, field, apogee_id):
    return f"{config.ASPCAP_DIR}/{telescope}/{quote(field)}/aspcapStar-dr17-{quote(apogee_id)}.fits"


def fetch_spectrum(telescope, field, apogee_id):
    """Renvoie (flux, err) en float32, ou None si le serveur n'a pas de fichier aspcapStar."""
    try:
        with _open(spectrum_url(telescope, field, apogee_id)) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    with fits.open(io.BytesIO(raw)) as hdul:
        return hdul[1].data.astype(np.float32), hdul[2].data.astype(np.float32)
