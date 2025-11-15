import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 📂 Définir les chemins de manière robuste
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "Data", "788438_data.csv")
model_path = os.path.join(current_dir, "..", "model.pkl")

print(f"📁 Chargement des données depuis: {data_path}")

try:
    # Charger les données
    df = pd.read_csv(data_path)
    print(f"✅ Données chargées: {df.shape}")
    
    # Vérifier que la colonne Revenue existe
    if "Revenue" not in df.columns:
        raise ValueError("❌ Colonne 'Revenue' non trouvée dans les données")
    
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    # 🔧 Prétraitement
    numeric_features = ["Marketing_Spend", "R&D_Spend", "Administration_Costs", "Number_of_Employees"]
    categorical_features = ["Region"]

    # Vérifier que les colonnes existent
    missing_num = [col for col in numeric_features if col not in df.columns]
    missing_cat = [col for col in categorical_features if col not in df.columns]
    
    if missing_num:
        raise ValueError(f"❌ Colonnes numériques manquantes: {missing_num}")
    if missing_cat:
        raise ValueError(f"❌ Colonnes catégorielles manquantes: {missing_cat}")

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ])

    # 🧠 Pipeline complet
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # 🎯 Entraînement
    print("🔄 Entraînement du modèle en cours...")
    model.fit(X, y)
    
    # 📊 Évaluation rapide
    score = model.score(X, y)
    print(f"📈 Score R² sur l'ensemble d'entraînement: {score:.4f}")

    # 💾 Sauvegarde
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé dans: {model_path}")

except FileNotFoundError:
    print(f"❌ Fichier données non trouvé: {data_path}")
    print("💡 Vérifiez le chemin du fichier CSV")
except Exception as e:
    print(f"❌ Erreur lors de l'entraînement: {e}")