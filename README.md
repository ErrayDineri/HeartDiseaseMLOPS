# Heart Disease ML Studio

Plateforme complète (frontend + backend) pour:

- explorer un dataset cardiovasculaire,
- entraîner plusieurs modèles de classification,
- tuner automatiquement les hyperparamètres,
- comparer les performances (Accuracy/F1/AUC, ROC/PR, matrice de confusion),
- versionner les datasets/modèles et suivre les expérimentations.

Ce README couvre **la partie projet** (architecture, exécution, API) et **la partie théorie** (ML supervisé, tuning, AutoML, MLOps).

---

## 1) Objectif du projet

Le projet implémente un workflow de bout en bout pour la classification binaire (présence/absence de maladie cardiaque):

1. Chargement/contrôle qualité de données,
2. Prétraitement automatique,
3. Entraînement multi-modèles,
4. Tuning (Grid/Random/Optuna),
5. Sélection du meilleur modèle,
6. Persistance + versioning,
7. Visualisation et exploitation via UI.

---

## 2) Structure du repository

- `data.csv` : dataset principal
- `eda.py` : EDA (statistiques + graphes)
- `data_quality_check.py` : audit qualité dataset
- `data_quality_report.md` / `data_quality_report.json` : rapports générés
- `backend/` : API FastAPI (entraînement, tuning, AutoML, MLOps)
- `frontend/` : interface Next.js connectée au backend
- `requirements.txt` : dépendances Python

---

## 3) Architecture technique

### Frontend (Next.js)

- Dashboard interactif responsive
- Sélection d’algorithmes et édition hyperparamètres
- Déclenchement des endpoints backend (`/train`, `/tune`, `/automl`)
- Visualisation des résultats (bar charts, ROC/PR, confusion matrix)
- Upload dataset, nettoyage rapide, export PNG/CSV, rollback modèle

### Backend (FastAPI)

- Services datasets: upload, preview, nettoyage, version active
- Services ML: training multi-modèles, tuning, AutoML, prédiction
- Persistance locale: modèles `joblib`, registres JSON datasets/modèles/expériences
- Documentation API auto via Swagger/ReDoc

### Stockage

Le backend utilise `backend/storage/` pour stocker:

- datasets versionnés,
- modèles entraînés,
- registres (JSON) des expérimentations et versions.

---

## 4) Setup et exécution

## Prérequis

- Python 3.10+
- Node.js 18+
- npm

## 4.1 Environnement Python

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 4.2 Lancer le backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 4.3 Lancer le frontend

```bash
cd frontend
npm install
npm run dev
```

Puis ouvrir `http://localhost:3000`.

Option URL backend personnalisée:

```bash
set NEXT_PUBLIC_API_BASE=http://localhost:8000
npm run dev
```

---

## 5) Contrôle qualité des données

Exécuter:

```bash
python data_quality_check.py
```

Sorties:

- `data_quality_report.md`
- `data_quality_report.json`

Constat actuel sur le dataset fourni:

- 1025 lignes, 14 colonnes,
- 0 valeur manquante,
- classes cible plutôt équilibrées,
- **beaucoup de doublons** (important pour l’évaluation).

> Recommandation pratique: entraîner avec `drop_duplicates=true` pour limiter l’optimisme artificiel des métriques.

---

## 6) Fonctionnalités implémentées

### 6.1 Gestion des modèles

- SVM
- Random Forest
- KNN
- Logistic Regression
- Neural Network (MLP)

Supporte la sélection multiple et la comparaison côte à côte.

### 6.2 Tuning

- GridSearchCV
- RandomizedSearchCV
- Optuna (si installé)

### 6.3 AutoML

- Test automatique de plusieurs algorithmes
- Tuning rapide optionnel
- Leaderboard + meilleur modèle

### 6.4 MLOps local

- versioning dataset/modèle
- historique d’expériences
- rollback vers version modèle précédente
- export/téléchargement d’artefacts

---

## 7) API backend (résumé)

### Santé

- `GET /health`

### Algorithmes

- `GET /algorithms`

### Datasets

- `GET /datasets`
- `POST /datasets/upload`
- `GET /datasets/{version}/preview`
- `POST /datasets/clean`

### Modélisation

- `POST /train`
- `POST /tune`
- `POST /automl`

### MLOps

- `GET /experiments`
- `GET /models`
- `POST /models/{model_version}/rollback`
- `GET /models/{model_version}/download`
- `POST /models/{model_version}/predict`

Exemple minimal de requête training:

```bash
curl -X POST http://localhost:8000/train \
	-H "Content-Type: application/json" \
	-d "{\"dataset_version\":\"heart_v1\",\"models\":[\"svm\",\"random_forest\"],\"target_column\":\"target\",\"drop_duplicates\":true}"
```

---

## 8) Théorie ML (essentiel)

## 8.1 Problème supervisé

On cherche une fonction $f(X)$ qui prédit une cible binaire $y \in \{0,1\}$ à partir de variables cliniques (âge, cholestérol, etc.).

Pipeline standard:

1. split train/test,
2. prétraitement,
3. entraînement,
4. évaluation,
5. sélection/itération.

## 8.2 Prétraitement

- Variables numériques: imputation médiane + standardisation.
- Variables catégorielles: imputation mode + One-Hot Encoding.

Pourquoi?

- éviter les erreurs dues aux données manquantes,
- rendre les échelles comparables (important pour SVM/KNN/MLP),
- transformer le catégoriel en format exploitable par les modèles.

## 8.3 Modèles utilisés

### SVM

- Cherche une frontière séparatrice à marge maximale.
- Très sensible à `C` et `gamma` (si noyau RBF).

### Random Forest

- Ensemble d’arbres de décision (bagging + random feature sampling).
- Robuste, souvent bon baseline avancé.

### KNN

- Prédit selon les voisins les plus proches.
- Sensible au choix de `k` et à l’échelle des variables.

### Logistic Regression

- Modèle linéaire probabiliste pour classification binaire.
- Interprétable, bon baseline.

### Neural Network (MLP)

- Réseau dense capable de capturer des relations non-linéaires.
- Demande tuning (architecture, learning rate, itérations).

## 8.4 Métriques

- **Accuracy**: proportion de bonnes prédictions.
- **Precision**: parmi les positifs prédits, part des vrais positifs.
- **Recall**: parmi les vrais positifs, part correctement détectée.
- **F1-score**: compromis precision/recall.
- **ROC-AUC**: capacité de séparation globale des classes.

Rappel utile:

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

## 8.5 Tuning des hyperparamètres

Les hyperparamètres ne sont pas appris automatiquement pendant le fit (ex: `max_depth`, `C`, `k`).

- **GridSearch**: teste toutes les combinaisons définies (exhaustif, coûteux).
- **RandomSearch**: teste des combinaisons tirées aléatoirement (plus rapide).
- **Optuna**: optimisation bayésienne/heuristique (souvent plus efficace pour grands espaces).

Objectif: maximiser une métrique de validation croisée (souvent F1 pour classes déséquilibrées).

## 8.6 AutoML

AutoML automatise:

1. sélection d’algorithmes,
2. tuning hyperparamètres,
3. ranking des candidats,
4. proposition du meilleur modèle.

Gain: accélère l’itération. Limite: ne remplace pas l’analyse métier et la validation rigoureuse.

## 8.7 MLOps (niveau projet)

Dans ce projet, la couche MLOps locale assure:

- traçabilité (dataset version + model version + métriques),
- reproductibilité des runs,
- historique d’expériences,
- rollback rapide.

---

## 9) Bonnes pratiques recommandées

1. Dédupliquer le dataset avant comparaison finale.
2. Utiliser une métrique alignée au besoin clinique (souvent Recall/F1).
3. Garder un test set jamais vu pendant le tuning.
4. Comparer plusieurs seeds/partitions (stabilité).
5. Documenter systématiquement hyperparamètres + versions.

---

## 10) Limites actuelles

- Tracking local JSON (pas de serveur MLflow/DVC branché nativement).
- Pas d’orchestration asynchrone des jobs longs.
- Pas d’authentification API.
- Validation clinique/réglementaire hors scope.

---

## 11) Roadmap possible

- Intégration MLflow réelle (tracking UI + artifacts + model registry).
- Pipeline asynchrone (Celery/RQ + queue + progress réel).
- Authentification + rôles.
- Tests automatiques backend/frontend (CI).
- Monitoring drift/performance en production.

---

## 12) Dépannage rapide

### `ModuleNotFoundError: backend`

Lancer `uvicorn` depuis la racine du projet:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Front non connecté au backend

- Vérifier que l’API répond sur `http://localhost:8000/health`.
- Vérifier `NEXT_PUBLIC_API_BASE` si URL custom.

### Build frontend KO

```bash
cd frontend
npm install
npm run build
```

---

## 13) Avertissement important

Ce projet est un outil pédagogique/technique de ML. Il ne constitue pas un dispositif médical et ne doit pas être utilisé pour une décision clinique réelle sans validation médicale, statistique et réglementaire appropriée.
