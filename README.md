# 📈 Prédiction du chiffre d'affaires d'une entreprise

Ce projet utilise la régression linéaire pour prédire le chiffre d'affaires d'une entreprise à partir de ses dépenses et de sa localisation géographique.

## 🧠 Objectif

Créer un système interactif capable de :
- Charger et prétraiter un jeu de données
- Entraîner un modèle de régression linéaire
- Évaluer les performances du modèle
- Permettre à l'utilisateur de saisir des données pour prédire les revenus
- Fournir une interface web avec Streamlit

## 📂 Structure du projet
/StudDoc/DataScience/ │ ├── Données/               # Contient le fichier data.csv ├── Cahiers/               # Notebooks Jupyter ├── Scripts/               # Scripts Python (modèle, interface) ├── README.md              # Ce fichier ├── requirements.txt       # Dépendances du projet

## 📊 Jeu de données

Le fichier `data.csv` contient les colonnes suivantes :
- `Marketing_Spend`
- `R&D_Spend`
- `Administration_Costs`
- `Number_of_Employees`
- `Region` (Amérique du Nord, Europe, Asie)
- `Revenue` (variable cible)

## ⚙️ Prétraitement

- **Encodage à chaud** de la variable `Region`
- **Standardisation** des variables numériques

## 🧪 Évaluation du modèle

Le modèle est évalué avec :
- MAE (Erreur absolue moyenne)
- RMSE (Erreur quadratique moyenne)
- R² (Coefficient de détermination)

## 🖥️ Interface utilisateur

Deux versions :
- **Console interactive** : menu avec saisie utilisateur
- **Interface web** : développée avec Streamlit

## 🚀 Lancer l'application Streamlit

```bash
streamlit run app.py

📱 Accès mobile
Lancer avec :
streamlit run app.py --server.address=0.0.0.0

Puis accéder via navigateur mobile à l’adresse IP locale.

📌 Auteur
Projet réalisé par Koffi Modeste Konan





