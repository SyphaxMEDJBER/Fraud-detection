"""Vérifie que evaluate_model() calcule des métriques valides, sur un modèle jouet minuscule."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.evaluate import evaluate_model


def test_evaluate_model_renvoie_des_metriques_entre_0_et_1():
    # Données jouets minuscules, juste pour avoir un modèle entraîné en une fraction
    # de seconde (pas besoin des vraies 450000 lignes pour tester evaluate_model).
    X = np.array([[0], [1], [2], [3], [10], [11], [12], [13]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    metrics = evaluate_model(model, X, y)

    for nom in ["precision", "recall", "roc_auc", "pr_auc"]:
        assert 0.0 <= metrics[nom] <= 1.0
