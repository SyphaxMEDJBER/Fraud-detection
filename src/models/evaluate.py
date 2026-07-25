"""Évaluation d'un modèle avec des métriques adaptées à un dataset déséquilibré (pas l'accuracy)."""

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # predict_proba renvoie 2 colonnes : proba d'être classe 0, proba d'être classe 1 (fraude).
    # On garde uniquement la colonne 1 : ROC-AUC et PR-AUC ont besoin d'un score continu
    # (à quel point le modèle est "confiant" que c'est une fraude), pas juste 0/1.
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Matrice de confusion (lignes = réel, colonnes = prédit) :")
    print("           prédit=0   prédit=1")
    matrix = confusion_matrix(y_test, y_pred)
    print(f"réel=0     {matrix[0][0]:>8}   {matrix[0][1]:>8}")
    print(f"réel=1     {matrix[1][0]:>8}   {matrix[1][1]:>8}")

    print(f"\nPrecision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR-AUC    : {average_precision_score(y_test, y_proba):.4f}")
