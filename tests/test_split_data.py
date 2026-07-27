"""Vérifie que load_train_test() sépare bien les données sans en perdre, et garde le déséquilibre."""

from sqlalchemy import create_engine, text

from src.data.split_data import DB_PATH, TABLE_NAME, load_train_test

TOLERANCE = 0.01  # marge d'erreur acceptée, le split n'est jamais pile exact


def _total_et_pourcentage_fraude_reels():
    # Interroge directement la table pour connaître le VRAI total et le VRAI %
    # de fraude, quelle que soit la base utilisée (le vrai dataset en local,
    # ou le petit échantillon de test en CI) -> le test reste valable dans les 2 cas.
    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.connect() as connexion:
        total = connexion.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
        fraudes = connexion.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE Class = 1")).scalar()
    return total, fraudes / total


def test_split_garde_toutes_les_lignes():
    total_attendu, _ = _total_et_pourcentage_fraude_reels()
    X_train, X_test, y_train, y_test = load_train_test()

    assert len(X_train) + len(X_test) == total_attendu


def test_split_garde_le_desequilibre_grace_au_stratify():
    _, pourcentage_attendu = _total_et_pourcentage_fraude_reels()
    X_train, X_test, y_train, y_test = load_train_test()

    pourcentage_train = y_train.mean()
    pourcentage_test = y_test.mean()

    # abs(a - b) < TOLERANCE : vérifie que les deux pourcentages sont proches du
    # pourcentage attendu, sans exiger une égalité parfaite (les nombres calculés
    # sur des sous-groupes différents ne tombent jamais exactement pile au même chiffre).
    assert abs(pourcentage_train - pourcentage_attendu) < TOLERANCE
    assert abs(pourcentage_test - pourcentage_attendu) < TOLERANCE
