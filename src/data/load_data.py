"""Charge le CSV brut dans une base SQLite pour permettre l'exploration en SQL."""

from pathlib import Path  # gère les chemins de façon portable (Windows/Linux/Mac)

import pandas as pd
from sqlalchemy import create_engine

RAW_CSV_PATH = Path("data/raw/creditcard.csv")
DB_PATH = Path("data/processed/fraud.db")
TABLE_NAME = "transactions"


def load_csv_to_sqlite(csv_path: Path = RAW_CSV_PATH, db_path: Path = DB_PATH) -> None:
    df = pd.read_csv(csv_path)

    # sqlite:///<chemin> = même API SQLAlchemy que pour Postgres/MySQL en prod,
    # seule cette ligne changerait si on migrait vers un vrai serveur SQL.
    engine = create_engine(f"sqlite:///{db_path}")

    # if_exists="replace" : recrée la table à chaque exécution pour que le script
    # soit rejouable sans dupliquer les lignes (idempotence).
    # index=False : l'index pandas (0,1,2...) n'a pas de sens métier, on ne le stocke pas.
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

    print(f"{len(df)} lignes chargées dans {db_path} (table '{TABLE_NAME}')")


# Ne s'exécute que si on lance ce fichier directement (python src/data/load_data.py),
# pas si on l'importe depuis un notebook ou un test.
if __name__ == "__main__":
    load_csv_to_sqlite()
