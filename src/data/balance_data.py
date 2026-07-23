"""Rééquilibre le train set avec SMOTE (jamais le test set, cf. split_data.py)."""

from imblearn.over_sampling import SMOTE

from src.data.split_data import load_train_test

RANDOM_STATE = 42  # même valeur que dans split_data.py, pour la reproductibilité


def apply_smote(X_train, y_train):
    # SMOTE crée des exemples synthétiques de fraude en s'appuyant sur les fraudes
    # réelles voisines dans le train set (par défaut, k_neighbors=5 fraudes voisines
    # servent de base pour générer chaque nouveau point).
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    return X_train_balanced, y_train_balanced


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_train_test()

    print(f"Avant SMOTE : {len(X_train)} lignes, {y_train.mean() * 100:.4f}% de fraudes")

    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    print(f"Après SMOTE : {len(X_train_balanced)} lignes, {y_train_balanced.mean() * 100:.4f}% de fraudes")

    # Rappel : le test set n'est jamais touché par SMOTE, il garde sa vraie
    # proportion de fraudes (~0,17%) pour une évaluation réaliste.
    print(f"Test (inchangé) : {len(X_test)} lignes, {y_test.mean() * 100:.4f}% de fraudes")
