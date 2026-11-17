# 10_label_source.py

## Rôle

Sépare les erreurs des CNN de régression selon l'origine de l'étiquette (`aspcap` ou `fparam`) et la qualité du spectre. L'analyse montre d'où vient l'erreur sur le test complet.

## Commande

```bash
python scripts/10_label_source.py
python scripts/10_label_source.py --runs cnn_regress_good
```

| Option | Défaut | Effet |
|---|---|---|
| `--runs` | `cnn_regress_good cnn_regress_all` | runs de régression à évaluer |

## Sortie

`label_source.csv` : une ligne par run, origine d'étiquette et qualité, avec l'erreur absolue moyenne sur Teff, log g et [M/H].

## Lecture

Une erreur forte sur les étoiles `fparam` signale des étiquettes peu fiables, pas forcément un modèle faible : FPARAM n'est pas calibré.
