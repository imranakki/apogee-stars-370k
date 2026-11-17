#!/usr/bin/env bash
# Expérience qualité : chaque modèle entraîné sur les bons spectres, puis sur tous.
# Relancer le script reprend les runs inachevés grâce aux checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

for train_on in good all; do
  for task in class regress; do
    python scripts/05_baseline.py --task "$task" --train-on "$train_on"
    python scripts/06_train.py --model cnn --task "$task" --train-on "$train_on"
  done
  python scripts/06_train.py --model autoencoder --train-on "$train_on"
  python scripts/07_cluster.py --features pca --subset "$train_on"
  python scripts/07_cluster.py --features latent --ae-run "autoencoder_none_$train_on" --subset "$train_on"
done

python scripts/08_report.py
python scripts/09_quality_by_snr.py
