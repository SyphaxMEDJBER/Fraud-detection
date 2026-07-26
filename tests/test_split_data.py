"""Vérifie que load_train_test() sépare bien les données sans en perdre, et garde le déséquilibre."""

from src.data.split_data import load_train_test

TOTAL_LIGNES = 284807
POURCENTAGE_FRAUDE_ATTENDU = 0.001727  # ~0,1727%, vu pendant l'EDA
TOLERANCE = 0.0005  # marge d'erreur acceptée, le split n'est jamais pile exact


def test_split_garde_toutes_les_lignes():
    X_train, X_test, y_train, y_test = load_train_test()

    assert len(X_train) + len(X_test) == TOTAL_LIGNES


def test_split_garde_le_desequilibre_grace_au_stratify():
    X_train, X_test, y_train, y_test = load_train_test()

    pourcentage_train = y_train.mean()
    pourcentage_test = y_test.mean()

    # abs(a - b) < TOLERANCE : vérifie que les deux pourcentages sont proches du
    # pourcentage attendu, sans exiger une égalité parfaite (les nombres calculés
    # sur des sous-groupes différents ne tombent jamais exactement pile au même chiffre).
    assert abs(pourcentage_train - POURCENTAGE_FRAUDE_ATTENDU) < TOLERANCE
    assert abs(pourcentage_test - POURCENTAGE_FRAUDE_ATTENDU) < TOLERANCE
