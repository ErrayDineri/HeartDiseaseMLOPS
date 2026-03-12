# Heart Disease ML Studio

A platform for:
- Exploring cardiovascular datasets
- Training and tuning classification models
- Comparing performance metrics (Accuracy, F1, AUC, etc.)
- Versioning datasets/models and tracking experiments

## Project Workflow
1. Data loading and quality checks
2. Automatic preprocessing
3. Multi-model training
4. Hyperparameter tuning (Grid/Random/Optuna)
5. Best model selection
6. Persistence and versioning
7. Visualization via UI

## Repository Structure
- `data.csv`: Main dataset
- `eda.py`: Exploratory Data Analysis
- `backend/`: FastAPI backend for training, tuning, and MLOps
- `frontend/`: Next.js interface for visualization
- `requirements.txt`: Python dependencies

## Technical Overview
### Frontend
- Interactive dashboard
- Algorithm selection and hyperparameter editing
- Visualization of results (ROC, PR, confusion matrix)
- Dataset upload and quick cleaning

### Backend
- API for dataset management, training, tuning, and AutoML
- MLflow integration for experiment tracking
- Supported algorithms: SVM, Random Forest, KNN, Logistic Regression, Neural Network
