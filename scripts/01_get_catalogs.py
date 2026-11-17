"""Télécharge allStar (4 Go), le catalogue red clump et le catalogue OCCAM dans STARS_DATA."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stars import config
from stars.download import download_file


def main():
    config.DATA.mkdir(parents=True, exist_ok=True)
    for dest, url in config.URLS.items():
        print(f"{dest.name} ...", flush=True)
        download_file(url, dest)
    print(f"Catalogues prêts dans {config.DATA}")


if __name__ == "__main__":
    main()
