# Déployer PyroWaste sur Streamlit Community Cloud

Après publication du dépôt privé GitHub :

1. Ouvrir https://share.streamlit.io/ et se connecter avec le compte GitHub Zack137.
2. Autoriser Streamlit à accéder au dépôt privé `Zack137/pyrowaste`.
3. Cliquer sur **Create app**, puis choisir le déploiement depuis GitHub.
4. Renseigner :
   - Repository : `Zack137/pyrowaste`
   - Branch : `main`
   - Main file path : `app.py`
5. Dans **Advanced settings**, sélectionner **Python 3.11**.
6. Cliquer sur **Deploy** et attendre la fin de l'installation.

Les dépendances sont déclarées dans `requirements.txt`. Les modèles entraînés
sont inclus et vérifiés par empreinte SHA-256 des données : aucun entraînement
n'est nécessaire au premier démarrage avec les fichiers fournis.

Le dépôt privé ne nécessite pas d'être rendu public pour déployer l'application.
La visibilité de l'application se règle dans les paramètres de Streamlit.

Documentation officielle : https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
