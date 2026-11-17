# 09_quality_by_snr.py

## Rôle

Mesure la performance des CNN par tranche de SNR sur le test complet. Cette analyse montre où les mauvais spectres changent le résultat, ce que les moyennes globales cachent.

Le script recharge `best.pt` de chaque run, prédit le test complet (`test_all`) et calcule les métriques dans quatre tranches de SNR : 5 à 30, 30 à 70, 70 à 150, plus de 150. Une tranche avec moins de 20 étoiles est ignorée.

## Commande

```bash
python scripts/09_quality_by_snr.py
python scripts/09_quality_by_snr.py --runs cnn_class_good cnn_class_all
```

| Option | Défaut | Effet |
|---|---|---|
| `--runs` | les 4 runs CNN | runs à évaluer, ils doivent exister dans `runs/` |

## Sorties

- `quality_by_snr.csv` : une ligne par run et par tranche ;
- `figures/quality/macro_f1_by_snr.png`, `mae_LOGG_by_snr.png`, `mae_TEFF_by_snr.png`.

## Lecture

Comparez les courbes `*_good` et `*_all`. Un écart dans la tranche 5 à 30 et aucun écart au-dessus de 70 signifie que les spectres bruités à l'entraînement aident seulement sur les données bruitées.
