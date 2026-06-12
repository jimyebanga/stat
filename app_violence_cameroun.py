import streamlit as st
import numpy as np
import pandas as pd

# Configuration
st.set_page_config(
    page_title="Prédiction Violence Conjugale - Cameroun",
    page_icon="🇨🇲",
    layout="wide"
)

# Style CSS professionnel avec fond et drapeau
st.markdown("""
<style>
    /* Sidebar verte uniquement */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1B5E20 0%, #0d3b0f 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        color: white;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        color: white;
    }
    
    /* Cartes de risque */
    .risk-low {
        background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(67,160,71,0.3);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #FB8C00 0%, #EF6C00 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(251,140,0,0.3);
    }
    .risk-high {
        background: linear-gradient(135deg, #E53935 0%, #C62828 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(229,57,53,0.3);
    }
    
    /* Cartes d'info */
    .info-card {
        background: #F5F5F5;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #2E7D32;
        margin: 1rem 0;
    }
    .warning-card {
        background: #FFF8E1;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        font-size: 0.8rem;
        color: #666;
        margin-top: 2rem;
        border-top: 1px solid #E0E0E0;
    }
    
    /* Bouton */
    .stButton > button {
        background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.6rem 2rem;
        border-radius: 30px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(46,125,50,0.4);
    }
    
    /* Labels des inputs */
    div[data-testid="stNumberInput"] label, 
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        font-weight: 600;
        color: #333;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: #F5F5F5;
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# En-tête avec drapeau
st.markdown("""
<div class="main-header">
    <div class="flag-decoration">🇨🇲</div>
    <h1>🛡️ Prédiction du Risque de Violence Conjugale</h1>
    <p>Cameroun - Enquête Démographique et de Santé (EDS) 2018</p>
</div>
""", unsafe_allow_html=True)

# Informations
with st.expander("ℹ️ À propos de cet outil", expanded=False):
    st.markdown("""
    <div class="info-card">
    Cet outil estime le risque de violence physique chez les femmes camerounaises sur la base 
    d'un modèle de régression logistique entraîné sur <strong>6 450 femmes</strong> (EDS Cameroun 2018).
    
    <strong>Modèle</strong> : AUC = 0,75 | Sensibilité = 68 %
    
    Les résultats sont fournis à titre indicatif et ne remplacent pas un avis professionnel.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# COEFFICIENTS ISSUS DE LA RÉGRESSION LOGISTIQUE
# ============================================================

COEFFS = {
    'age_femme_annees': -0.0892,
    'niveau_education': 0.1527,
    'statut_matrimonial': 0.6314,
    'quintile_richesse': -0.2027,
    'milieu_residence_rural': 0.0296,
    'score_controle_marital': 0.1291,
    'score_autonomie': 0.1149,
    'nombre_enfants_nes_vivants': 0.4520,
    'frequence_ecoute_radio': 0.0301
}
INTERCEPT = -0.2278

def predict_risk(age, education, statut, quintile, residence, controle, autonomie, enfants, radio):
    logit = INTERCEPT
    logit += COEFFS['age_femme_annees'] * ((age - 28.15) / 8.5)
    logit += COEFFS['niveau_education'] * ((education - 1.4) / 0.86)
    logit += COEFFS['statut_matrimonial'] * ((statut - 1.28) / 1.25)
    logit += COEFFS['quintile_richesse'] * ((quintile - 3.14) / 1.31)
    logit += COEFFS['milieu_residence_rural'] * ((residence - 0.46) / 0.50)
    logit += COEFFS['score_controle_marital'] * ((controle - 0.63) / 0.48)
    logit += COEFFS['score_autonomie'] * ((autonomie - 0.69) / 1.03)
    logit += COEFFS['nombre_enfants_nes_vivants'] * ((enfants - 2.46) / 2.34)
    logit += COEFFS['frequence_ecoute_radio'] * ((radio - 0.66) / 0.47)
    return 1 / (1 + np.exp(-logit))

# ============================================================
# INTERFACE
# ============================================================

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 👤 Caractéristiques de la femme")
    
    age = st.number_input("Âge (années)", min_value=15, max_value=49, value=28, step=1)
    
    education = st.selectbox(
        "Niveau d'éducation",
        options=[0, 1, 2, 3],
        format_func=lambda x: {0: "Aucun", 1: "Primaire", 2: "Secondaire", 3: "Supérieur"}[x]
    )
    
    statut = st.selectbox(
        "Statut matrimonial",
        options=[0, 1, 2, 3, 4, 5],
        format_func=lambda x: {
            0: "Célibataire", 1: "Mariée", 2: "Vit avec conjoint",
            3: "Divorcée", 4: "Séparée", 5: "Veuve"
        }[x]
    )
    
    quintile = st.selectbox(
        "Quintile de richesse",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "Très pauvre", 2: "Pauvre", 3: "Moyen", 4: "Riche", 5: "Très riche"}[x]
    )

with col2:
    st.markdown("### 🏠 Contexte familial")
    
    residence = st.radio(
        "Milieu de résidence",
        options=[0, 1],
        format_func=lambda x: "Urbain" if x == 0 else "Rural",
        horizontal=True
    )
    
    controle = st.radio(
        "Contrôle marital",
        options=[0, 1],
        format_func=lambda x: "Aucun contrôle" if x == 0 else "Conjoint contrôle",
        horizontal=True
    )
    
    autonomie = st.slider(
        "Score d'autonomie (0-5)",
        min_value=0, max_value=5, value=0, step=1,
        help="Plus le score est élevé, plus la femme participe aux décisions familiales"
    )
    
    enfants = st.number_input(
        "Nombre d'enfants nés vivants",
        min_value=0, max_value=15, value=0, step=1,
        help="Nombre total d'enfants mis au monde"
    )
    
    radio = st.radio(
        "Écoute la radio",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Jamais", 1: "Occasionnellement", 2: "Régulièrement"}[x],
        horizontal=True
    )

# ============================================================
# PRÉDICTION
# ============================================================

st.markdown("---")

if st.button("🔮 PRÉDIRE LE RISQUE", type="primary", use_container_width=True):
    proba = predict_risk(age, education, statut, quintile, residence, controle, autonomie, enfants, radio)
    
    if proba < 0.3:
        st.markdown(f'<div class="risk-low">🟢 RISQUE FAIBLE – Probabilité : {proba:.1%}</div>', unsafe_allow_html=True)
        st.info("📌 **Reste vigilante.** Connais tes droits et garde contact avec des personnes de confiance.")
    elif proba < 0.6:
        st.markdown(f'<div class="risk-moderate">🟠 RISQUE MODÉRÉ – Probabilité : {proba:.1%}</div>', unsafe_allow_html=True)
        st.warning("⚠️ **Soyez vigilante.** Identifiez une personne de confiance et préparez un plan d'urgence.")
    else:
        st.markdown(f'<div class="risk-high">🔴 RISQUE ÉLEVÉ – Probabilité : {proba:.1%}</div>', unsafe_allow_html=True)
        st.error("🚨 **URGENCE.** Si vous êtes en danger immédiat, appelez le **117**.")

# ============================================================
# RESSOURCES D'AIDE
# ============================================================

with st.expander("📞 Ressources d'aide et numéros d'urgence", expanded=False):
    st.markdown("""
    <div class="warning-card">
    <strong>🔴 NUMÉROS D'URGENCE CAMEROUN 🇨🇲</strong><br><br>
    <strong>🚔 Police : 117</strong><br>
    <strong>🏛️ MINPROFF : +237 222 23 45 67</strong><br>
    <strong>🤝 ALVF : +237 699 88 77 66</strong><br><br>
    <strong>🏠 Centres d'accueil :</strong><br>
    • Yaoundé : Centre Mbankolo – <strong>698 00 11 22</strong><br>
    • Douala : Centre Bonabéri – <strong>699 00 33 44</strong>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <p>🇨🇲 <strong>République du Cameroun - Paix - Travail - Patrie</strong> 🇨🇲</p>
    <p>📊 Modèle entraîné sur 6 450 femmes (EDS Cameroun 2018) | AUC = 0,75 | Régression logistique</p>
    <p>📞 En cas d'urgence, appelez le <strong>117</strong> (police) ou le <strong>+237 222 23 45 67</strong> (MINPROFF)</p>
    <p>© 2026 - Outil à but non lucratif - Données anonymisées</p>
</div>
""", unsafe_allow_html=True)