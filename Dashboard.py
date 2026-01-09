import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 🖼️ Configuration de la page
st.set_page_config(
    page_title="Prédiction du chiffre d'affaires", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border: none;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #4a5568 0%, #2d3748 100%);
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">📊 Prédiction du Chiffre d\'Affaires Entreprise</h1>', unsafe_allow_html=True)
st.markdown("Entrez les données de l'entreprise pour estimer le revenu prédit avec notre modèle d'IA.")

# 📍 Chemins robustes
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir,  "model.pkl")
data_path = os.path.join(current_dir,  "Data", "788438_data.csv")

# 🔍 Chargement des données et modèle avec gestion d'erreurs
@st.cache_data
def load_data():
    try:
        if not os.path.exists(data_path):
            st.error(f"❌ Données non trouvées à: {data_path}")
            return None
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des données: {e}")
        return None

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Modèle non trouvé à: {model_path}")
            st.info("💡 Exécutez d'abord `train_model.py` pour créer le modèle")
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

# Sidebar pour les informations système
with st.sidebar:
    st.header("🔧 Configuration Système")
    
    # Chargement avec indicateurs de progression
    with st.spinner("Chargement du modèle..."):
        model = load_model()
    if model:
        st.success("✅ Modèle chargé")
    
    with st.spinner("Chargement des données..."):
        df = load_data()
    if df is not None:
        st.success("✅ Données chargées")
        st.info(f"📁 {df.shape[0]} entreprises | {df.shape[1]} variables")

# Calcul des métriques si les données sont disponibles
if df is not None and model is not None:
    try:
        X = df.drop("Revenue", axis=1)
        y = df["Revenue"]
        y_pred = model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        avg_revenue = df["Revenue"].mean()
        
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul des métriques: {e}")
        st.stop()

# 🎯 Interface principale
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🧮 Saisie des Données Entreprise")
    
    # Formulaire de saisie
    with st.form("prediction_form"):
        st.write("**Informations Financières**")
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            marketing = st.number_input(
                "💰 Budget Marketing (€)", 
                min_value=0.0, 
                step=1000.0,
                value=100000.0,
                help="Investissement total en marketing"
            )
            rnd = st.number_input(
                "🔬 R&D Spend (€)", 
                min_value=0.0, 
                step=1000.0,
                value=150000.0,
                help="Budget recherche et développement"
            )
        
        with col1b:
            admin = st.number_input(
                "🏢 Coûts Administratifs (€)", 
                min_value=0.0, 
                step=1000.0,
                value=120000.0,
                help="Frais administratifs totaux"
            )
            employees = st.number_input(
                "👥 Effectif Salariés", 
                min_value=0, 
                step=1,
                value=100,
                help="Nombre total d'employés"
            )
        
        st.write("**Informations Géographiques**")
        region = st.selectbox(
            "🌍 Région", 
            ["North America", "Europe", "Asia"],
            help="Région géographique du siège"
        )
        
        # Bouton de soumission
        submitted = st.form_submit_button("🎯 Prédire le Chiffre d'Affaires", type="primary")

with col2:
    st.subheader("📊 Performance du Modèle")
    
    if df is not None and model is not None:
        # Métriques de performance
        col2a, col2b, col2c = st.columns(3)
        
        with col2a:
            st.markdown(f"""
            <div class="metric-card">
                <h3>MAE</h3>
                <h2>{mae:,.0f} €</h2>
                <p>Erreur Absolue Moyenne</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2b:
            st.markdown(f"""
            <div class="metric-card">
                <h3>RMSE</h3>
                <h2>{rmse:,.0f} €</h2>
                <p>Racine Erreur Quadratique</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2c:
            st.markdown(f"""
            <div class="metric-card">
                <h3>R²</h3>
                <h2>{r2:.3f}</h2>
                <p>Score de Détermination</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Statistiques descriptives
        with st.expander("📈 Statistiques des Données", expanded=True):
            st.write(f"**Chiffre d'affaires moyen:** {avg_revenue:,.2f} €")
            st.write(f"**Plage des données:** {df['Revenue'].min():,.0f} € - {df['Revenue'].max():,.0f} €")
            st.write(f"**Nombre d'entreprises:** {len(df)}")

# 🔮 Traitement de la prédiction
if submitted and model is not None:
    try:
        # Préparation des données
        user_input = pd.DataFrame([{
            "Marketing_Spend": marketing,
            "R&D_Spend": rnd,
            "Administration_Costs": admin,
            "Number_of_Employees": employees,
            "Region": region
        }])
        
        # Prédiction
        prediction = model.predict(user_input)[0]
        
        # Affichage des résultats
        st.markdown("---")
        
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 PRÉDICTION TERMINÉE</h2>
                <h1>{prediction:,.2f} €</h1>
                <p>Chiffre d'Affaires Annuel Estimé</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_res2:
            if df is not None:
                difference = prediction - avg_revenue
                percentage_diff = (difference / avg_revenue) * 100
                
                st.metric(
                    label="📊 Comparaison avec la moyenne",
                    value=f"{prediction:,.0f} €",
                    delta=f"{difference:,.0f} € ({percentage_diff:+.1f}%)"
                )
        
        # Analyse détaillée
        with st.expander("🔍 Analyse Détaillée", expanded=True):
            col_ana1, col_ana2, col_ana3 = st.columns(3)
            
            with col_ana1:
                st.metric("Marketing/CA", f"{(marketing/prediction*100):.1f}%" if prediction > 0 else "N/A")
            with col_ana2:
                st.metric("R&D/CA", f"{(rnd/prediction*100):.1f}%" if prediction > 0 else "N/A")
            with col_ana3:
                st.metric("Admin/CA", f"{(admin/prediction*100):.1f}%" if prediction > 0 else "N/A")
                
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction: {e}")

# 📋 Section informations supplémentaires
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📖 Instructions", "🔍 Données", "ℹ️ À Propos"])

with tab1:
    st.header("Guide d'Utilisation")
    st.markdown("""
    ### 🎯 Comment obtenir une prédiction :
    1. **Renseignez les données financières** dans le formulaire de gauche
    2. **Sélectionnez la région** de l'entreprise
    3. **Cliquez sur le bouton de prédiction** pour lancer l'analyse
    
    ### 📊 Interprétation des résultats :
    - **MAE** : Erreur moyenne absolue en euros (plus bas = mieux)
    - **RMSE** : Erreur quadratique moyenne (penalise les grosses erreurs)
    - **R²** : Pourcentage de variance expliquée (0-1, plus haut = mieux)
    """)

with tab2:
    st.header("Aperçu des Données d'Entraînement")
    if df is not None:
        col_data1, col_data2 = st.columns([2, 1])
        
        with col_data1:
            st.write("**Échantillon des données :**")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col_data2:
            st.write("**Variables disponibles :**")
            for col in df.columns:
                st.write(f"- {col}")

with tab3:
    st.header("Informations Techniques")
    st.markdown("""
    ### 🚀 Fonctionnalités :
    - **Modèle de Machine Learning** : Régression avancée
    - **Prétraitement automatique** : Normalisation et encodage
    - **Interface responsive** : Adapté à tous devices
    
    ### 🔧 Stack Technique :
    - **Framework** : Streamlit
    - **ML** : Scikit-learn
    - **Data** : Pandas, NumPy
    - **Style** : CSS personnalisé
    
    ### 📈 Méthodologie :
    - Entraînement sur données historiques
    - Validation croisée
    - Métriques de performance robustes
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "© 2025 Prédiction CA Entreprise | Insight by Koffi.ds "
    "</div>", 
    unsafe_allow_html=True
)