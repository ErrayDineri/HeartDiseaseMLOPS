# Frontend - Heart Disease ML Studio

Next.js interface for:
- Multi-model selection and training
- Hyperparameter tuning (GridSearch, RandomSearch, Optuna)
- Interactive visualizations (confusion matrix, ROC, PR)
- Dataset upload, cleaning, and preview
- MLOps tracking (versions, experiments, rollback)

## Setup
```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Backend Connection
- Default: `http://localhost:8000`
- To change:
```bash
set NEXT_PUBLIC_API_BASE=http://new-backend-url
npm run dev
```
