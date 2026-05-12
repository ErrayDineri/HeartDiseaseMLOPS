# Analyse Random Forest (HeartDisease)

## Objectif
Ce document resume l'analyse Random Forest demandee (importance des variables, stabilite, erreurs, biais/variance) a partir des artefacts generes par le script.

## Resultats principaux
- Top 3 variables (importance agregee par variable): cp (0.1524), ca (0.1173), thalach (0.1140).
- Stabilite sur 10 seeds: accuracy moyenne 0.9966 (ecart-type 0.0049), F1 moyenne 0.9967 (ecart-type 0.0047).
- Erreurs: 2 exemples mal classes avec confiance ~0.616 (voir details dans le JSON).
- Biais/variance: avec max_depth=5, train ~0.94-0.95 et test ~0.93-0.94; pour max_depth=10 ou None, train/test = 1.00 (suggerant un jeu tres facile ou une possible fuite a verifier).

## Artefacts generes
- Script: [rf_analysis.py](rf_analysis.py)
- Figure importance: [rf_feature_importance.png](rf_feature_importance.png)
- Grille biais/variance: [rf_bias_variance.csv](rf_bias_variance.csv)
- Resume detaille: [rf_analysis_summary.json](rf_analysis_summary.json)

## Reproduire l'analyse
Execution simple apres activation du venv:

```bash
python rf_analysis.py
```

Les fichiers sont regeneres dans la racine du projet.

## Interpretation rapide
- `cp`, `ca` et `thalach` dominent la prediction.
- La stabilite est excellente sur les seeds testes.
- La perfectibilite sur certaines configurations (train/test a 1.00) merite une verification des doublons ou d'une fuite cible.
