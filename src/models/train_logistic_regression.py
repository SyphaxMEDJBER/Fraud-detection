"""Entraîne une Logistic Regression sur le train rééquilibré par SMOTE, évalue sur le vrai test set."""

import mlflow
from sklearn.linear_model import LogisticRegression

from src.data.balance_data import apply_smote
from src.data.split_data import load_train_test
from src.models.evaluate import evaluate_model

RANDOM_STATE = 42
MAX_ITER = 1000


def train_logistic_regression():
    X_train, X_test, y_train, y_test = load_train_test()
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # max_iter=1000 : la valeur par défaut (100) ne suffit souvent pas à faire converger
    # le modèle avec 30 variables et ~450000 lignes après SMOTE -> on lui laisse plus
    # d'itérations pour trouver la meilleure frontière de décision.
    model = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)
    model.fit(X_train_balanced, y_train_balanced)

    return model, X_test, y_test


if __name__ == "__main__":
    # start_run() ouvre une nouvelle fiche MLflow. run_name : le nom affiché dans
    # l'interface, pour reconnaître ce run parmi tous les autres.
    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", MAX_ITER)

        model, X_test, y_test = train_logistic_regression()
        evaluate_model(model, X_test, y_test)
