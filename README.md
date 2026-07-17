# Détection de fraude bancaire — Pipeline ML + Agent RAG

Projet de détection de fraude sur transactions par carte bancaire, à partir du dataset
[Credit Card Fraud Detection (MLG-ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Objectifs
- Pipeline de traitement de données structuré (ingestion, transformation, stockage SQL)
- Comparaison de modèles de classification (Logistic Regression, Random Forest, XGBoost)
- Gestion du déséquilibre de classes (SMOTE)
- Évaluation avec métriques adaptées (recall, precision, ROC-AUC, PR-AUC)
- Tracking des expériences avec MLflow
- Agent RAG pour interroger les résultats du modèle en langage naturel

## Statut
🚧 En cours de développement

## Structure du projet
Voir l'arborescence du repo.

## Installation
```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
```
