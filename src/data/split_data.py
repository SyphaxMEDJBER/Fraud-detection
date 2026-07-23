"""Sépare les données en train/test AVANT tout rééquilibrage (SMOTE viendra après, sur le train uniquement)."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text

DB_PATH = Path("data/processed/fraud.db")
TABLE_NAME = "transactions"
TARGET_COL = "Class"

# 20% en test : assez de données pour un vrai "examen blanc" fiable, tout en gardant
# 80% pour réviser (entraîner le modèle).
TEST_SIZE = 0.2
RANDOM_STATE = 42  # fixe la génération aléatoire -> même split à chaque exécution (reproductible)


def load_train_test(db_path: Path = DB_PATH, test_size: float = TEST_SIZE):
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql(text(f"SELECT * FROM {TABLE_NAME}"), engine)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # stratify=y : force le split à garder la même proportion de fraudes (0,17%) dans le
    # train ET dans le test. Sans ça, un split purement aléatoire pourrait par malchance
    # envoyer trop (ou pas assez) de fraudes d'un seul côté, vu qu'elles sont très rares.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_train_test()

    print(f"Train : {len(X_train)} lignes, {y_train.mean() * 100:.4f}% de fraudes")
    print(f"Test  : {len(X_test)} lignes, {y_test.mean() * 100:.4f}% de fraudes")
