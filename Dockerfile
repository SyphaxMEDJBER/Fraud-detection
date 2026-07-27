# Image de base : Python 3.10 en version "slim" (allégée, moins de composants inutiles
# que l'image Python complète -> image finale plus légère et plus rapide à construire).
FROM python:3.10-slim

# Dossier de travail à l'intérieur du container -> toutes les commandes suivantes
# s'exécutent depuis là (équivalent d'un "cd /app").
WORKDIR /app

# On copie d'abord SEULEMENT requirements.txt (pas encore tout le code) et on installe
# les librairies ici, avant de copier le reste. Cet ordre permet à Docker de "mettre
# en cache" cette étape : si seul le code change (pas les dépendances), Docker réutilise
# l'installation déjà faite au lieu de tout réinstaller -> builds bien plus rapides.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Maintenant on copie tout le reste du projet (code source, tests...).
COPY . .

# Commande lancée par défaut quand on démarre le container :
# 1. charge le CSV vers SQLite (fraud.db n'existe pas encore dans l'image, on le
#    régénère -> voir .dockerignore, on ne copie pas data/processed/)
# 2. lance les tests, pour prouver que tout fonctionne dans cet environnement isolé.
CMD ["sh", "-c", "python -m src.data.load_data && python -m pytest tests/ -v"]
