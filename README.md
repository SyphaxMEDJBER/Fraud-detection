# Détection de fraude bancaire

![CI](https://github.com/SyphaxMEDJBER/Fraud-detection/actions/workflows/ci.yml/badge.svg)

Pipeline complet de détection de fraude sur transactions par carte bancaire : exploration
SQL/pandas, gestion du déséquilibre de classes (SMOTE), comparaison de 3 modèles de
classification, tracking des expériences avec MLflow, tests automatisés, containerisation
Docker et pipeline CI/CD (GitHub Actions).

Dataset : [Credit Card Fraud Detection (MLG-ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284 807 transactions, dont 492 fraudes (0,17 %).

## Sommaire

- [Pipeline](#pipeline)
- [Résultats](#résultats)
- [Stack technique](#stack-technique)
- [Structure du projet](#structure-du-projet)
- [Installation et utilisation](#installation-et-utilisation)
- [Tests](#tests)
- [Docker](#docker)
- [CI/CD](#cicd)

## Pipeline

1. **Chargement & exploration** — le CSV est chargé dans une base SQLite (`src/data/load_data.py`),
   puis exploré en SQL (répartition des classes, valeurs manquantes, statistiques) et en pandas
   (visualisations) dans `notebooks/01_eda_sql.ipynb`.
2. **Gestion du déséquilibre** — split train/test stratifié (`src/data/split_data.py`), puis
   rééquilibrage du train uniquement via SMOTE (`src/data/balance_data.py`). Le test set reste
   intact pour une évaluation réaliste.
3. **Modélisation** — 3 modèles entraînés et comparés sur les mêmes données : Logistic Regression,
   Random Forest, XGBoost (`src/models/train_*.py`), évalués avec des métriques adaptées à un
   dataset déséquilibré (`src/models/evaluate.py`).
4. **Tracking** — chaque run (paramètres, métriques, modèle entraîné) est enregistré avec MLflow.
5. **Qualité & déploiement** — tests automatisés (pytest), containerisation (Docker), pipeline
   CI/CD (GitHub Actions) qui teste et publie une image Docker à chaque push sur `main`.

## Résultats

Modèles entraînés sur le train rééquilibré par SMOTE, évalués sur le vrai test set (98 fraudes
réelles sur 56 962 transactions) :

| Modèle | Recall | Precision | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 89,8 % | 12,5 % | 0,9746 | 0,7183 |
| Random Forest | 82,7 % | 83,5 % | 0,9644 | **0,8747** |
| XGBoost | 84,7 % | 79,1 % | **0,9831** | 0,8671 |

**Pourquoi PR-AUC plutôt qu'accuracy** : avec seulement 0,17 % de fraudes, un modèle qui prédit
systématiquement "pas de fraude" atteindrait 99,83 % d'accuracy sans jamais rien détecter. Le
PR-AUC, plus sévère sur les classes rares, donne une image plus honnête de la performance.

**Random Forest et XGBoost surpassent nettement la Logistic Regression** : cette dernière détecte
le plus de fraudes (Recall), mais au prix de 617 fausses alertes contre 16 et 22 respectivement —
inutilisable en pratique. Random Forest et XGBoost offrent le meilleur compromis
precision/recall.

## Stack technique

**Données & ML** : Python, pandas, SQLAlchemy (SQLite), scikit-learn, imbalanced-learn (SMOTE), XGBoost
**MLOps** : MLflow
**Qualité & déploiement** : pytest, Docker, GitHub Actions

## Structure du projet

```
├── data/
│   ├── raw/                  # Dataset brut (non versionné, voir Installation)
│   └── processed/            # Base SQLite générée (non versionnée)
├── notebooks/
│   └── 01_eda_sql.ipynb      # Exploration SQL + visualisations pandas
├── src/
│   ├── data/
│   │   ├── load_data.py      # CSV -> SQLite
│   │   ├── split_data.py     # Split train/test stratifié
│   │   └── balance_data.py   # SMOTE (train uniquement)
│   └── models/
│       ├── evaluate.py                  # Métriques + tracking MLflow
│       ├── train_logistic_regression.py
│       ├── train_random_forest.py
│       └── train_xgboost.py
├── tests/
│   ├── fixtures/sample_creditcard.csv   # Échantillon pour les tests (CI)
│   └── test_*.py
├── .github/workflows/ci.yml  # Pipeline CI/CD
├── Dockerfile
└── requirements.txt
```

## Installation et utilisation

```bash
git clone https://github.com/SyphaxMEDJBER/Fraud-detection.git
cd Fraud-detection
```

Dans les deux cas ci-dessous, le dataset brut (`creditcard.csv`, ~150 Mo) n'est pas inclus dans
le repo (restrictions de taille et de redistribution Kaggle) : télécharge-le depuis
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) et place-le dans `data/raw/`.

### Option rapide : Docker

Vérifie que le pipeline fonctionne, de façon reproductible, sans installer Python ni les
dépendances :
```bash
docker build -t fraud-detection .
docker run fraud-detection
```
Charge les données et exécute la suite de tests dans un environnement isolé (voir
[Docker](#docker)). Pour explorer le notebook, relancer un modèle en particulier ou utiliser
l'interface MLflow, utiliser plutôt l'installation manuelle ci-dessous.

### Installation manuelle (exploration interactive)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

**1. Charger les données dans SQLite**
```bash
python -m src.data.load_data
```

**2. Explorer les données**

Ouvrir `notebooks/01_eda_sql.ipynb` dans Jupyter ou VS Code.

**3. Entraîner et évaluer les modèles**
```bash
python -m src.models.train_logistic_regression
python -m src.models.train_random_forest
python -m src.models.train_xgboost
```

**4. Comparer les runs dans MLflow**
```bash
mlflow ui
```
Puis ouvrir [http://127.0.0.1:5000](http://127.0.0.1:5000) dans un navigateur.

## Tests

```bash
python -m pytest tests/ -v
```

Les tests utilisent un petit échantillon synthétique (`tests/fixtures/sample_creditcard.csv`)
plutôt que le vrai dataset — ils sont donc exécutables sans avoir téléchargé le CSV complet,
y compris en CI.

## Docker

```bash
docker build -t fraud-detection .
docker run fraud-detection
```

Le container charge les données et exécute la suite de tests dans un environnement isolé et
reproductible.

## CI/CD

À chaque push ou pull request sur `main`, GitHub Actions (`.github/workflows/ci.yml`) :

1. **CI** — installe les dépendances et exécute la suite de tests
2. **CD** — si les tests passent et que le push est sur `main`, construit et publie l'image
   Docker sur GitHub Container Registry (`ghcr.io/syphaxmedjber/fraud-detection`)
