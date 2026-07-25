"""Entraîne un Random Forest sur le train rééquilibré par SMOTE, évalue sur le vrai test set."""

from sklearn.ensemble import RandomForestClassifier

from src.data.balance_data import apply_smote
from src.data.split_data import load_train_test
from src.models.evaluate import evaluate_model

RANDOM_STATE = 42


def train_random_forest():
    X_train, X_test, y_train, y_test = load_train_test()
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # n_estimators=100 : le nombre d'arbres dans la "forêt". Chaque arbre vote, et le
    # résultat final est la majorité des votes -> plus d'arbres = décision plus stable,
    # au prix d'un entraînement plus long. 100 est la valeur par défaut, un bon compromis.
    # n_jobs=-1 : utilise tous les cœurs du processeur pour entraîner les arbres en
    # parallèle (contrairement à la Logistic Regression, les arbres sont indépendants).
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_balanced, y_train_balanced)

    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = train_random_forest()
    evaluate_model(model, X_test, y_test)
