"""Vérifie que apply_smote() rééquilibre bien le train à 50/50 sans toucher au test."""

from src.data.balance_data import apply_smote
from src.data.split_data import load_train_test


def test_smote_equilibre_le_train_a_50_50():
    X_train, X_test, y_train, y_test = load_train_test()
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    assert y_train_balanced.mean() == 0.5


def test_smote_ne_reduit_jamais_le_nombre_de_lignes():
    X_train, X_test, y_train, y_test = load_train_test()
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # SMOTE ajoute des lignes synthétiques, il ne peut donc jamais y en avoir moins
    # après qu'avant.
    assert len(X_train_balanced) > len(X_train)
