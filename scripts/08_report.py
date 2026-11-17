"""Rassemble les metrics.json de tous les runs dans deux tableaux : supervisé et clustering."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from stars import config


def flatten(run, metrics):
    row = {"run": run}
    for key, value in metrics.items():
        if key == "run":
            continue
        if isinstance(value, dict):
            row.update({f"{key}.{k}": v for k, v in value.items()})
        else:
            row[key] = value
    return row


def main():
    supervised, clustering = [], []
    for path in sorted(config.RUNS.glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        row = flatten(path.parent.name, metrics)
        if path.parent.name.startswith("cluster_"):
            clustering.append(row)
        elif "test_good" in metrics:
            # Les autoencodeurs n'ont pas de score de test, ils sont évalués par 07_cluster.
            supervised.append(row)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for name, rows in [("supervised", supervised), ("clustering", clustering)]:
        if not rows:
            print(f"Aucun run pour {name}")
            continue
        table = pd.DataFrame(rows)
        csv_path = config.DATA / f"summary_{name}.csv"
        table.to_csv(csv_path, index=False)
        print(f"\n{name} ({csv_path})")
        cols = ["run"] + [c for c in table.columns if c.split(".")[-1] in
                          {"accuracy", "macro_f1", "mae_TEFF", "mae_LOGG", "mae_M_H", "ari_class", "nmi_class", "occam_purity"}]
        print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
