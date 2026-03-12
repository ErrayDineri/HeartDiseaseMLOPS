# Frontend - Heart Disease ML Studio

Cette interface Next.js couvre les fonctionnalités demandées:

- Sélection multi-modèles (SVM, RF, KNN, Logistic Regression, Neural Network)
- Documentation courte de chaque algorithme
- Mode entraînement depuis zéro / pré-entraîné
- Hyperparamètres modifiables
- Tuning automatique (GridSearch / RandomSearch / Optuna)
- Sauvegarde/chargement de configurations
- Visualisations interactives (confusion, ROC, PR, comparaison performances)
- Export PNG des graphiques et CSV des résultats
- Preview dataset, sélection colonnes/classes, upload CSV, nettoyage rapide
- Suivi MLOps (versions dataset/modèle, historique expériences, rollback)
- Notifications de progression et tutoriel rapide
- Bonus AutoML via backend

## Démarrage

```bash
npm install
npm run dev
```

Ouvrir ensuite `http://localhost:3000`.

## Connexion backend FastAPI

Par défaut, le frontend appelle `http://localhost:8000`.

Optionnel (si backend sur autre URL):

```bash
set NEXT_PUBLIC_API_BASE=http://localhost:8000
npm run dev
```

## Notes

- Le frontend est connecté au backend FastAPI pour l’entraînement réel et le suivi des expériences.
- Les visualisations se mettent à jour avec les résultats renvoyés par l’API (`/train`, `/tune`, `/automl`).
