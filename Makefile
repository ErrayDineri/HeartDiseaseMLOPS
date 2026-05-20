SHELL := /bin/sh

.PHONY: help backend frontend mlflow install-backend install-frontend install

help:
	@echo "Targets:"
	@echo "  backend          Run FastAPI backend (expects venv activated)"
	@echo "  frontend         Run Next.js frontend"
	@echo "  mlflow           Run MLflow UI"
	@echo "  install-backend  Install Python dependencies"
	@echo "  install-frontend Install Node dependencies"
	@echo "  install          Install backend and frontend dependencies"

backend:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend; npm install; npm run dev

mlflow:
	python -m mlflow ui --backend-store-uri sqlite:///backend/mlflow.db --host 127.0.0.1 --port 5000

install-backend:
	pip install -r requirements.txt

install-frontend:
	cd frontend; npm install

install: install-backend install-frontend
