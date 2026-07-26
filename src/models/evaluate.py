"""Évaluation d'un modèle avec des métriques adaptées à un dataset déséquilibré (pas l'accuracy)."""

import mlflow
import mlflow.sklearn
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

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print("Matrice de confusion (lignes = réel, colonnes = prédit) :")
    print("           prédit=0   prédit=1")
    matrix = confusion_matrix(y_test, y_pred)
    print(f"réel=0     {matrix[0][0]:>8}   {matrix[0][1]:>8}")
    print(f"réel=1     {matrix[1][0]:>8}   {matrix[1][1]:>8}")

    print(f"\nPrecision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    # On écrit les résultats ("la note du plat") sur la fiche MLflow ouverte par le
    # script appelant, plus le modèle entraîné lui-même pour pouvoir le réutiliser
    # plus tard sans le réentraîner.
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.sklearn.log_model(model, "model")

    # Renvoyé en plus de l'affichage : permet de tester la fonction (vérifier les
    # valeurs) sans avoir à parser le texte affiché dans le terminal.
    return {"precision": precision, "recall": recall, "roc_auc": roc_auc, "pr_auc": pr_auc}
