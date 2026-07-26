"""Entraîne un XGBoost sur le train rééquilibré par SMOTE, évalue sur le vrai test set."""

from xgboost import XGBClassifier

from src.data.balance_data import apply_smote
from src.data.split_data import load_train_test
from src.models.evaluate import evaluate_model

RANDOM_STATE = 42


def train_xgboost():
    X_train, X_test, y_train, y_test = load_train_test()
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # n_estimators=100 : le nombre d'arbres, comme pour Random Forest. Mais ici, contrairement
    # à Random Forest, les arbres ne sont pas indépendants : l'Arbre n°2 est construit pour
    # corriger les erreurs de l'Arbre n°1, l'Arbre n°3 corrige ce qu'il reste après les 2
    # premiers, etc. (boosting = construction séquentielle, chaque arbre corrige le précédent).
    model = XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric="logloss")
    model.fit(X_train_balanced, y_train_balanced)

    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = train_xgboost()
    evaluate_model(model, X_test, y_test)
