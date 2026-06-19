import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
import pickle
import os
from PIL import Image
import requests
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de primo-nuptialité - Cameroun",
    page_icon="🇨🇲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    /* Couleurs du Cameroun */
    :root {
        --cameroon-green: #007A5E;
        --cameroon-red: #CE1126;
        --cameroon-yellow: #FCD116;
    }
    
    .main-header {
        background: linear-gradient(135deg, #007A5E 0%, #007A5E 33%, #FCD116 33%, #FCD116 66%, #CE1126 66%, #CE1126 100%);
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .main-title {
        background-color: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .main-title h1 {
        color: #1a1a2e;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-title p {
        color: #4a4a6a;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #007A5E;
    }
    
    .card-red {
        border-left-color: #CE1126;
    }
    
    .card-yellow {
        border-left-color: #FCD116;
    }
    
    .result-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
    }
    
    .prob-high {
        background-color: #007A5E;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .prob-moderate {
        background-color: #FCD116;
        color: #1a1a2e;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .prob-low {
        background-color: #CE1126;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #007A5E 0%, #005a44 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(135deg, #008a6a 0%, #007A5E 100%);
        color: white;
    }
    
    .sidebar-title {
        color: #007A5E;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #6c757d;
        border-top: 2px solid #e9ecef;
        margin-top: 2rem;
    }
    
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-bottom: 3px solid #007A5E;
    }
    
    .stat-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .stat-card .label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Chargement du modèle et des données
@st.cache_resource
def load_model_and_data():
    model_path = "gbs_model.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model, scaler, feature_columns = pickle.load(f)
        return model, scaler, feature_columns
    else:
        return train_model()

@st.cache_data
def train_model():
    data = pd.read_csv('primo_nuptialite_clean.csv')
    data = data.dropna()
    
    cat_cols = ['milieu', 'instruction', 'richesse', 'region', 'religion']
    df_ml = data.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    
    X = df_ml.drop(columns=['duree', 'union'])
    y = Surv.from_dataframe('union', 'duree', df_ml)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingSurvivalAnalysis(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    with open("gbs_model.pkl", 'wb') as f:
        pickle.dump((model, scaler, X.columns.tolist()), f)
    
    return model, scaler, X.columns.tolist()

@st.cache_data
def load_data():
    data = pd.read_csv('primo_nuptialite_clean.csv')
    return data

# Fonction pour créer le graphique de survie
def plot_survival_curve(surv_func, title="Courbe de survie prédite"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Style du graphique
    ax.step(surv_func.x, surv_func(surv_func.x), where='post', 
            color='#007A5E', linewidth=3, label='Probabilité de survie')
    ax.fill_between(surv_func.x, 0, surv_func(surv_func.x), 
                     alpha=0.25, color='#007A5E')
    
    # Ajout de la ligne de référence à 0.5
    ax.axhline(y=0.5, color='#CE1126', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(surv_func.x[-1]*0.9, 0.52, 'Référence 0.5', color='#CE1126', fontsize=10)
    
    # Personnalisation
    ax.set_title(title, fontsize=18, fontweight='bold', color='#1a1a2e')
    ax.set_xlabel("Âge (années)", fontsize=13, fontweight='500')
    ax.set_ylabel("S(t) - Probabilité de ne pas être en union", fontsize=13, fontweight='500')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 55)
    
    # Bordures plus propres
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

# Fonction principale
def main():
    # En-tête avec bandeau aux couleurs du Cameroun
    st.markdown("""
    <div class="main-header">
        <div class="main-title">
            <h1>Prédiction de primo-nuptialité au Cameroun</h1>
            <p>Modèle Gradient Boosting Survival Analysis - EDS Cameroun 2018</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données et du modèle
    with st.spinner("Chargement du modèle en cours..."):
        model, scaler, feature_columns = load_model_and_data()
        data = load_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Caractéristiques de l\'individu</p>', unsafe_allow_html=True)
        st.markdown("---")
        
        age = st.slider("Âge actuel", 15, 55, 25, 1, 
                        help="Âge de l'individu au moment de l'enquête")
        
        st.markdown("---")
        
        milieu = st.selectbox(
            "Milieu de résidence",
            ["Urbain", "Rural"],
            help="Zone de résidence principale"
        )
        
        instruction = st.selectbox(
            "Niveau d'instruction",
            ["Aucun", "Primaire", "Secondaire", "Supérieur"],
            help="Plus haut niveau d'éducation atteint"
        )
        
        region = st.selectbox(
            "Région",
            ["Adamaoua", "Centre", "Est", "Extrême-Nord", "Littoral", 
             "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest"],
            help="Région administrative de résidence"
        )
        
        religion = st.selectbox(
            "Religion",
            ["Catholique", "Protestant", "Musulman", "Animiste", "Autre"],
            help="Appartenance religieuse"
        )
        
        richesse = st.selectbox(
            "Niveau de richesse",
            ["Pauvre", "Moyen", "Riche"],
            help="Quintile de richesse du ménage"
        )
        
        st.markdown("---")
        predict_button = st.button("Prédire la survie", use_container_width=True)
    
    # Statistiques générales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="value">{:,}</div>
            <div class="label">Individus dans la base</div>
        </div>
        """.format(len(data)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="value">{:.1%}</div>
            <div class="label">Taux d'union</div>
        </div>
        """.format(data['union'].mean()), unsafe_allow_html=True)
    
    with col3:
        age_moyen = data[data['union']==1]['duree'].mean()
        st.markdown("""
        <div class="stat-card">
            <div class="value">{:.1f} ans</div>
            <div class="label">Âge moyen à l'union</div>
        </div>
        """.format(age_moyen), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prédiction
    if predict_button:
        # Préparation des données
        input_data = {
            'age_actuel': age,
            'milieu': 0 if milieu == "Urbain" else 1,
            'instruction': ["Aucun", "Primaire", "Secondaire", "Supérieur"].index(instruction),
            'region': ["Adamaoua", "Centre", "Est", "Extrême-Nord", "Littoral", 
                       "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest"].index(region),
            'religion': ["Catholique", "Protestant", "Musulman", "Animiste", "Autre"].index(religion),
            'richesse': ["Pauvre", "Moyen", "Riche"].index(richesse)
        }
        
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        # Prédiction
        surv_func = model.predict_survival_function(input_scaled)[0]
        
        # Calcul des probabilités
        def get_prob(age_val):
            if age_val in surv_func.x:
                return surv_func(age_val)
            elif any(surv_func.x <= age_val):
                return surv_func(surv_func.x[surv_func.x <= age_val][-1])
            else:
                return 1.0
        
        prob_25 = get_prob(25)
        prob_30 = get_prob(30)
        prob_35 = get_prob(35)
        prob_40 = get_prob(40)
        
        # Affichage des résultats
        st.markdown("## Résultats de la prédiction")
        
        # Cartes de résultats
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        with col_res1:
            st.markdown("""
            <div class="stat-card">
                <div class="value" style="font-size:1.4rem;">{:.1%}</div>
                <div class="label">Probabilité à 25 ans</div>
            </div>
            """.format(prob_25), unsafe_allow_html=True)
        with col_res2:
            st.markdown("""
            <div class="stat-card" style="border-bottom-color: #FCD116;">
                <div class="value" style="font-size:1.4rem;">{:.1%}</div>
                <div class="label">Probabilité à 30 ans</div>
            </div>
            """.format(prob_30), unsafe_allow_html=True)
        with col_res3:
            st.markdown("""
            <div class="stat-card" style="border-bottom-color: #CE1126;">
                <div class="value" style="font-size:1.4rem;">{:.1%}</div>
                <div class="label">Probabilité à 35 ans</div>
            </div>
            """.format(prob_35), unsafe_allow_html=True)
        with col_res4:
            st.markdown("""
            <div class="stat-card" style="border-bottom-color: #6c757d;">
                <div class="value" style="font-size:1.4rem;">{:.1%}</div>
                <div class="label">Probabilité à 40 ans</div>
            </div>
            """.format(prob_40), unsafe_allow_html=True)
        
        # Interprétation
        if prob_30 > 0.5:
            prob_class = "prob-high"
            interpretation = "Probabilité élevée de ne pas être en union à 30 ans"
        elif prob_30 > 0.25:
            prob_class = "prob-moderate"
            interpretation = "Probabilité modérée de ne pas être en union à 30 ans"
        else:
            prob_class = "prob-low"
            interpretation = "Probabilité faible de ne pas être en union à 30 ans"
        
        st.markdown(f"""
        <div class="result-box" style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <span style="font-size: 1.1rem; font-weight: 500;">Interprétation :</span>
                <span class="{prob_class}">{interpretation}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Courbe de survie
        st.markdown("### Courbe de survie prédite")
        fig = plot_survival_curve(
            surv_func, 
            "Courbe de survie - {} - {} - {}".format(milieu, instruction, region)
        )
        st.pyplot(fig)
        
        # Détails
        with st.expander("Détails de la prédiction"):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Caractéristiques de l'individu :**")
                st.markdown("""
                - **Milieu :** {}  
                - **Instruction :** {}  
                - **Région :** {}  
                - **Religion :** {}  
                - **Niveau de richesse :** {}
                """.format(milieu, instruction, region, religion, richesse))
            
            with col_d2:
                st.markdown("**Probabilités de survie :**")
                probs_text = ""
                for age_plot in [18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50]:
                    prob = get_prob(age_plot)
                    probs_text += "- À {} ans : {:.1%}\n".format(age_plot, prob)
                st.markdown(probs_text)
    
    # Informations sur le modèle
    with st.expander("À propos du modèle"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            **Modèle : Gradient Boosting Survival Analysis**
            
            Ce modèle prédit la probabilité qu'un individu reste célibataire (non en union) 
            en fonction de ses caractéristiques démographiques et socio-économiques.
            
            **Variables utilisées :**
            - Âge actuel
            - Milieu de résidence (Urbain/Rural)
            - Niveau d'instruction
            - Région
            - Religion
            - Niveau de richesse
            """)
        with col_m2:
            st.markdown("""
            **Performance du modèle :**
            
            | Métrique | Valeur |
            |----------|--------|
            | C-index (Train) | 0.6824 |
            | C-index (Test) | 0.6762 |
            
            **Source des données :**
            Enquête Démographique et de Santé (EDS) Cameroun 2018
            """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Développé pour l'analyse de la primo-nuptialité au Cameroun</p>
        <p style="font-size: 0.8rem; color: #adb5bd;">Données EDS Cameroun 2018 - Modèle Gradient Boosting Survival</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()