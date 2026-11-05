# Atelier MLOps - Git + DVC + MLflow

Classification Iris : Random Forest et KNN.

## Installation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
DVC s'installe a part : pipx install dvc

## Lancer
python src/prepare_data.py
dvc pull
dvc repro
mlflow ui --backend-store-uri sqlite:///mlflow.db
