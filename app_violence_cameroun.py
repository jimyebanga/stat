"""
=============================================================================
  Application Streamlit – Déterminants de la Violence Physique chez les Femmes
  EDS Cameroun 2018 | Meilleur modèle : Random Forest (AUC = 0.7425)
=============================================================================
  Lancement :  streamlit run app_violence_physique.py
  Dépendances: pip install streamlit pandas numpy matplotlib seaborn
               scikit-learn shap joblib
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
from io import BytesIO

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG GLOBALE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Violence Physique – EDS Cameroun 2018",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette couleurs
C_ROUGE   = "#EF5350"
C_VERT    = "#66BB6A"
C_BLEU    = "#1565C0"
C_ORANGE  = "#F57F17"
C_GRIS    = "#90A4AE"
C_VIOLET  = "#6A1B9A"

RANDOM_STATE = 42

# Encodages identiques au NB4 (LabelEncoder sur modalités triées)
ENCODINGS = {
    "education_femme"   : ["Aucun", "Inconnu", "Primaire", "Secondaire", "Superieur"],
    "sit_matrimoniale"  : ["Divorcee", "Inconnu", "Jamais_en_union", "Mariee",
                           "Separee", "Union_libre", "Veuve"],
    "enceinte"          : ["Inconnu", "Non", "Oui"],
    "lecture_journal"   : ["Au_moins_1x_sem", "Inconnu", "Jamais",
                           "Moins_1x_sem", "Presque_chaque_jour"],
    "ecoute_radio"      : ["Au_moins_1x_sem", "Inconnu", "Jamais",
                           "Moins_1x_sem", "Presque_chaque_jour"],
    "regarde_tv"        : ["Au_moins_1x_sem", "Inconnu", "Jamais",
                           "Moins_1x_sem", "Presque_chaque_jour"],
    "education_conjoint": ["Aucun", "Inconnu", "Primaire", "Secondaire", "Superieur"],
    "conjoint_alcool"   : ["Inconnu", "Non", "Oui"],
    "region"            : ["Adamaoua", "Centre", "Est", "Extreme_Nord",
                           "Inconnu", "Littoral", "Nord", "Nord_Ouest",
                           "Ouest", "Sud", "Sud_Ouest"],
    "milieu"            : ["Inconnu", "Rural", "Urbain"],
}

FEATURES = [
    "age_femme", "education_femme", "sit_matrimoniale", "enceinte",
    "lecture_journal", "ecoute_radio", "regarde_tv", "score_media",
    "age_conjoint", "ecart_age", "education_conjoint", "conjoint_alcool",
    "region", "milieu"
]

FEATURES_FR = {
    "age_femme"          : "Âge de la femme",
    "education_femme"    : "Éducation (femme)",
    "sit_matrimoniale"   : "Situation matrimoniale",
    "enceinte"           : "Grossesse actuelle",
    "lecture_journal"    : "Lecture journaux",
    "ecoute_radio"       : "Écoute radio",
    "regarde_tv"         : "Télévision",
    "score_media"        : "Score médias (0–3)",
    "age_conjoint"       : "Âge du conjoint",
    "ecart_age"          : "Écart d'âge (conj.–femme)",
    "education_conjoint" : "Éducation (conjoint)",
    "conjoint_alcool"    : "Conjoint & alcool",
    "region"             : "Région",
    "milieu"             : "Milieu de résidence",
}

# ─────────────────────────────────────────────────────────────────────────────
#  FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduit l'encodage LabelEncoder du NB4."""
    df_enc = df.copy()
    for col, classes in ENCODINGS.items():
        if col in df_enc.columns:
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            df_enc[col] = df_enc[col].fillna("Inconnu").astype(str)
            df_enc[col] = df_enc[col].apply(
                lambda x: int(le.transform([x])[0]) if x in le.classes_
                else int(le.transform(["Inconnu"])[0])
            )
    return df_enc


@st.cache_resource(show_spinner=False)
def load_or_train_model(data_path: str):
    """Charge meilleur_modele.pkl si dispo, sinon réentraîne depuis le CSV."""
    if os.path.exists("meilleur_modele.pkl") and os.path.exists("data_split.pkl"):
        pipeline   = joblib.load("meilleur_modele.pkl")
        split_data = joblib.load("data_split.pkl")
        return pipeline, split_data

    if not os.path.exists(data_path):
        return None, None

    data = pd.read_csv(data_path)
    df_ml = data[FEATURES + ["violence_physique"]].dropna(subset=["violence_physique"]).copy()
    df_enc = encode_features(df_ml)

    X = df_enc[FEATURES].values
    y = df_enc["violence_physique"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ))
    ])
    pipeline.fit(X_train, y_train)

    split_data = {"X_train": X_train, "X_test": X_test,
                  "y_train": y_train, "y_test": y_test,
                  "features": FEATURES}
    joblib.dump(pipeline, "meilleur_modele.pkl")
    joblib.dump(split_data, "data_split.pkl")
    return pipeline, split_data


def gauge_chart(prob: float) -> plt.Figure:
    """Jauge de risque semi-circulaire."""
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.2, 1.3)
    ax.axis("off")

    zones = [(0.00, 0.33, C_VERT), (0.33, 0.55, C_ORANGE), (0.55, 1.00, C_ROUGE)]
    for lo, hi, col in zones:
        theta = np.linspace(np.pi * (1 - hi), np.pi * (1 - lo), 60)
        ax.fill_between(np.cos(theta), np.zeros_like(theta),
                        np.sin(theta), color=col, alpha=0.85)

    angle = np.pi * (1 - prob)
    ax.annotate("", xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
    ax.text(0, -0.15, f"{prob*100:.1f}%",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=C_ROUGE if prob > 0.55 else C_ORANGE if prob > 0.33 else C_VERT)
    ax.text(0, 0.9, "Probabilité de violence physique",
            ha="center", va="center", fontsize=8, color="gray")
    fig.patch.set_facecolor("none")
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  BARRE LATÉRALE – Navigation & données
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Flag_of_Cameroon.svg/320px-Flag_of_Cameroon.svg.png",
             width=80)
    st.markdown("## 🔬 Violence physique\n**EDS Cameroun 2018**")
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Accueil",
        "📊 Analyse descriptive",
        "📈 Performances des modèles",
        "🤖 Prédiction individuelle",
        "🔍 Explicabilité (SHAP)",
    ])
    st.divider()

    st.markdown("### 📂 Données")
    data_path = st.text_input("Chemin du CSV nettoyé",
                              value="violence_physique_clean.csv")
    uploaded = st.file_uploader("ou charger un fichier CSV", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df_raw.to_csv(data_path, index=False)
        st.success(f"Fichier chargé : {uploaded.name}")

    st.divider()
    st.caption("Modèle : Random Forest  \nAUC = 0.7425 | F1 = 0.5919")
    st.caption("📚 EDS Cameroun 2018 – CMIR71FL.dta")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 0 – ACCUEIL
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Accueil":
    st.title("🔬 Déterminants de la Violence Physique chez les Femmes")
    st.markdown("#### Enquête Démographique et de Santé – Cameroun 2018")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Femmes analysées", "4 690", help="Module violence domestique")
    col2.metric("Prévalence violence", "33.3%", help="D106 ou D107 = Oui")
    col3.metric("Meilleur modèle", "Random Forest", help="Sélectionné sur AUC")
    col4.metric("AUC-ROC", "0.7425", delta="+0.0216 vs logit")

    st.divider()
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("""
### Contexte
La violence physique exercée par le partenaire intime constitue un problème
de santé publique majeur au Cameroun. L'Enquête Démographique et de Santé 2018
(EDS-V) a collecté des données sur **4 690 femmes** éligibles au module
violence domestique.

Cette application permet :
- **Explorer** la distribution des facteurs de risque
- **Comparer** les performances de 6 modèles de classification
- **Prédire** le risque individuel de violence physique
- **Interpréter** le modèle via l'explicabilité SHAP
        """)

    with col_b:
        st.markdown("### Variables analysées")
        data_vars = {
            "Variable cible" : ["Violence physique (D106 + D107)"],
            "Femme"          : ["Âge", "Éducation", "Situation matrimoniale",
                                "Grossesse", "Exposition médias"],
            "Conjoint"       : ["Âge", "Éducation", "Consommation d'alcool"],
            "Contexte"       : ["Région (10)", "Milieu urbain/rural"],
        }
        for cat, items in data_vars.items():
            st.markdown(f"**{cat}**")
            for it in items:
                st.markdown(f"  - {it}")

    st.divider()
    st.markdown("### 🏆 Comparaison des 6 modèles (jeu de test)")
    df_perf = pd.DataFrame({
        "Modèle"    : ["Random Forest", "Gradient Boosting", "SVM",
                       "Arbre de Décision", "Régression Logistique", "KNN"],
        "AUC-ROC"   : [0.7425, 0.7330, 0.7250, 0.7232, 0.7209, 0.6993],
        "F1-score"  : [0.5919, 0.4981, 0.5965, 0.5700, 0.5957, 0.4457],
        "Précision" : [0.5117, 0.5929, 0.5103, 0.4597, 0.5091, 0.5493],
        "Rappel"    : [0.7019, 0.4295, 0.7179, 0.7500, 0.7179, 0.3750],
        "Accuracy"  : [0.6780, 0.7122, 0.6770, 0.6237, 0.6759, 0.6898],
    })

    def highlight_best(s):
        is_max = s == s.max()
        return ["background-color: #E8F5E9; font-weight: bold" if v else "" for v in is_max]

    styled = (df_perf.set_index("Modèle")
              .style
              .apply(highlight_best)
              .format("{:.4f}"))
    st.dataframe(styled, use_container_width=True)
    st.caption("🟢 Cellule en vert = meilleure valeur de la colonne")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 – ANALYSE DESCRIPTIVE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Analyse descriptive":
    st.title("📊 Analyse Descriptive")

    if not os.path.exists(data_path):
        st.warning("⚠ Fichier de données introuvable. Chargez `violence_physique_clean.csv`.")
        st.stop()

    data = load_data(data_path)
    st.success(f"Base chargée : {data.shape[0]:,} observations × {data.shape[1]} variables")

    # KPIs
    prevalence = data["violence_physique"].mean() * 100
    n_victimes = int(data["violence_physique"].sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("N total", f"{len(data):,}")
    col2.metric("Victimes", f"{n_victimes:,}")
    col3.metric("Prévalence", f"{prevalence:.1f}%")
    col4.metric("Non-victimes", f"{len(data) - n_victimes:,}")

    st.divider()
    tabs = st.tabs(["🎯 Cible", "👩 Femme", "👨 Conjoint", "🗺 Région & Milieu", "📰 Médias"])

    # ── Tab Cible ──
    with tabs[0]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        vc = data["violence_physique"].value_counts()
        axes[0].pie(vc.values, labels=["Non victime", "Victime"],
                    autopct="%1.1f%%", colors=[C_VERT, C_ROUGE], startangle=140,
                    wedgeprops={"edgecolor": "white", "linewidth": 2})
        axes[0].set_title("Prévalence de la violence physique")

        axes[1].hist(data[data["violence_physique"]==0]["age_femme"].dropna(),
                     alpha=0.65, bins=20, color=C_VERT, label="Non victime")
        axes[1].hist(data[data["violence_physique"]==1]["age_femme"].dropna(),
                     alpha=0.65, bins=20, color=C_ROUGE, label="Victime")
        axes[1].set_xlabel("Âge de la femme")
        axes[1].set_ylabel("Effectif")
        axes[1].set_title("Distribution de l'âge par statut")
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab Femme ──
    with tabs[1]:
        var_femme = st.selectbox("Variable", ["education_femme", "sit_matrimoniale", "enceinte"],
                                 format_func=lambda x: FEATURES_FR[x])
        prev = (data.groupby(var_femme)["violence_physique"]
                .mean().mul(100).sort_values(ascending=False))
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = [C_ROUGE if v > 35 else C_ORANGE if v > 25 else C_VERT for v in prev.values]
        bars = ax.bar(prev.index, prev.values, color=colors, edgecolor="white", width=0.6)
        for bar, v in zip(bars, prev.values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.axhline(prevalence, color="gray", ls="--", lw=1,
                   label=f"Prévalence globale ({prevalence:.1f}%)")
        ax.set_ylabel("Prévalence (%)")
        ax.set_title(f"Prévalence de la violence physique par {FEATURES_FR[var_femme]}")
        ax.set_ylim(0, prev.max() * 1.2)
        ax.legend()
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab Conjoint ──
    with tabs[2]:
        col_l, col_r = st.columns(2)
        with col_l:
            prev_alc = (data.groupby("conjoint_alcool")["violence_physique"]
                        .mean().mul(100).sort_values(ascending=False))
            fig, ax = plt.subplots(figsize=(5, 4))
            colors_alc = [C_ROUGE if v > prevalence else C_VERT for v in prev_alc.values]
            bars = ax.bar(prev_alc.index, prev_alc.values,
                          color=colors_alc, edgecolor="white", width=0.5)
            for bar, v in zip(bars, prev_alc.values):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%",
                        ha="center", fontsize=11, fontweight="bold")
            ax.axhline(prevalence, color="gray", ls="--", lw=1)
            ax.set_ylabel("Prévalence (%)")
            ax.set_title("Alcool du conjoint")
            ax.set_ylim(0, prev_alc.max() * 1.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_r:
            prev_educ_c = (data.groupby("education_conjoint")["violence_physique"]
                           .mean().mul(100).sort_values(ascending=False))
            fig, ax = plt.subplots(figsize=(5, 4))
            colors_ec = [C_ROUGE if v > prevalence else C_VERT for v in prev_educ_c.values]
            bars = ax.bar(prev_educ_c.index, prev_educ_c.values,
                          color=colors_ec, edgecolor="white", width=0.5)
            for bar, v in zip(bars, prev_educ_c.values):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%",
                        ha="center", fontsize=11, fontweight="bold")
            ax.axhline(prevalence, color="gray", ls="--", lw=1)
            ax.set_ylabel("Prévalence (%)")
            ax.set_title("Éducation du conjoint")
            ax.set_ylim(0, prev_educ_c.max() * 1.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Écart d'âge
        fig, ax = plt.subplots(figsize=(10, 4))
        for val, col, label in [(0, C_VERT, "Non victime"), (1, C_ROUGE, "Victime")]:
            ax.hist(data[data["violence_physique"]==val]["ecart_age"].dropna(),
                    bins=25, alpha=0.65, color=col, label=label, density=True)
        ax.set_xlabel("Écart d'âge (conjoint – femme)")
        ax.set_title("Distribution de l'écart d'âge par statut de violence")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab Région ──
    with tabs[3]:
        col_l, col_r = st.columns(2)
        with col_l:
            prev_reg = (data.groupby("region")["violence_physique"]
                        .agg(["mean", "count"])
                        .rename(columns={"mean": "prev", "count": "n"}))
            prev_reg["prev"] = prev_reg["prev"] * 100
            prev_reg = prev_reg.sort_values("prev", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            colors_r = [C_ROUGE if v > 35 else C_ORANGE if v > 25 else C_VERT
                        for v in prev_reg["prev"]]
            ax.barh(prev_reg.index, prev_reg["prev"],
                    color=colors_r, edgecolor="white", height=0.7)
            ax.axvline(prevalence, color="gray", ls="--", lw=1, label="Moy. nationale")
            for i, (reg, row) in enumerate(prev_reg.iterrows()):
                ax.text(row["prev"] + 0.3, i,
                        f"{row['prev']:.1f}% (n={int(row['n']):,})",
                        va="center", fontsize=9)
            ax.set_xlabel("Prévalence (%)")
            ax.set_title("Violence physique par région")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_r:
            prev_mil = (data.groupby("milieu")["violence_physique"]
                        .mean().mul(100).sort_values(ascending=False))
            fig, ax = plt.subplots(figsize=(4, 4))
            colors_m = [C_ROUGE if v > prevalence else C_VERT for v in prev_mil.values]
            bars = ax.bar(prev_mil.index, prev_mil.values,
                          color=colors_m, edgecolor="white", width=0.5)
            for bar, v in zip(bars, prev_mil.values):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}%",
                        ha="center", fontsize=13, fontweight="bold")
            ax.axhline(prevalence, color="gray", ls="--", lw=1)
            ax.set_ylabel("Prévalence (%)")
            ax.set_title("Urbain vs Rural")
            ax.set_ylim(0, prev_mil.max() * 1.25)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Tab Médias ──
    with tabs[4]:
        media_vars = ["lecture_journal", "ecoute_radio", "regarde_tv"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, mv in zip(axes, media_vars):
            prev_m = (data.groupby(mv)["violence_physique"]
                      .mean().mul(100).sort_values(ascending=False))
            colors_med = [C_ROUGE if v > prevalence else C_VERT for v in prev_m.values]
            bars = ax.bar(range(len(prev_m)), prev_m.values,
                          color=colors_med, edgecolor="white")
            ax.set_xticks(range(len(prev_m)))
            ax.set_xticklabels(prev_m.index, rotation=20, ha="right", fontsize=8)
            ax.axhline(prevalence, color="gray", ls="--", lw=1)
            ax.set_title(FEATURES_FR[mv])
            ax.set_ylabel("Prévalence (%)")
        plt.suptitle("Effet protecteur de l'exposition aux médias", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 – PERFORMANCES DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈 Performances des modèles":
    st.title("📈 Performances des Modèles")

    if not os.path.exists(data_path):
        st.warning("⚠ Fichier de données introuvable.")
        st.stop()

    with st.spinner("Chargement du modèle et des données..."):
        pipeline, split_data = load_or_train_model(data_path)

    if pipeline is None:
        st.error("Impossible de charger le modèle. Vérifiez le fichier de données.")
        st.stop()

    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_prob)
    f1  = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AUC-ROC",    f"{auc:.4f}")
    col2.metric("F1-score",   f"{f1:.4f}")
    col3.metric("Accuracy",   f"{acc:.4f}")
    col4.metric("Précision",  f"{prec:.4f}")
    col5.metric("Rappel",     f"{rec:.4f}")

    st.divider()
    tabs = st.tabs(["📉 Courbe ROC", "🧮 Matrice de confusion",
                    "📋 Rapport classification", "📚 Courbe d'apprentissage"])

    # ── ROC ──
    with tabs[0]:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.fill_between(fpr, tpr, alpha=0.15, color=C_BLEU)
        ax.plot(fpr, tpr, color=C_BLEU, lw=2.5,
                label=f"Random Forest (AUC = {auc:.3f})")
        ax.plot([0,1],[0,1], "k--", lw=1, label="Aléatoire (AUC = 0.5)")
        ax.set_xlabel("Taux de faux positifs (1 – Spécificité)")
        ax.set_ylabel("Taux de vrais positifs (Sensibilité)")
        ax.set_title("Courbe ROC – Random Forest")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.download_button("💾 Télécharger la figure",
                           data=fig_to_bytes(fig), file_name="roc_random_forest.png")

    # ── Confusion ──
    with tabs[1]:
        seuil = st.slider("Seuil de décision", 0.10, 0.90, 0.50, 0.05)
        y_pred_s = (y_pred_prob >= seuil).astype(int)
        cm = confusion_matrix(y_test, y_pred_s)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Non victime", "Victime"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Matrice de confusion (seuil = {seuil})")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        tn, fp, fn, tp = cm.ravel()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VP (Vrais Positifs)", tp)
        c2.metric("FP (Faux Positifs)", fp)
        c3.metric("FN (Faux Négatifs)", fn)
        c4.metric("VN (Vrais Négatifs)", tn)

    # ── Rapport ──
    with tabs[2]:
        report = classification_report(y_test, y_pred,
                                       target_names=["Non victime", "Victime"],
                                       output_dict=True)
        df_report = pd.DataFrame(report).T.drop("accuracy", errors="ignore")
        st.dataframe(df_report.style.format("{:.4f}").background_gradient(
            cmap="RdYlGn", subset=["precision", "recall", "f1-score"]),
            use_container_width=True)
        st.markdown("""
**Lecture :**
- **Précision** : parmi les femmes prédites victimes, proportion réellement victimes
- **Rappel** : parmi les vraies victimes, proportion bien identifiées par le modèle
- **F1-score** : moyenne harmonique précision / rappel
        """)

    # ── Courbe d'apprentissage ──
    with tabs[3]:
        st.info("Calcul en cours (validation croisée 5-fold)... environ 30 secondes.")
        with st.spinner("Calcul de la courbe d'apprentissage..."):
            X_train = split_data["X_train"]
            y_train = split_data["y_train"]
            train_sizes, train_scores, val_scores = learning_curve(
                pipeline, X_train, y_train,
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="roc_auc", n_jobs=-1
            )

        tr_mean = train_scores.mean(axis=1)
        tr_std  = train_scores.std(axis=1)
        va_mean = val_scores.mean(axis=1)
        va_std  = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(train_sizes, tr_mean, "o-", color=C_BLEU,   lw=2, label="Train")
        ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std,
                        alpha=0.15, color=C_BLEU)
        ax.plot(train_sizes, va_mean, "s-", color=C_ROUGE, lw=2, label="Validation")
        ax.fill_between(train_sizes, va_mean - va_std, va_mean + va_std,
                        alpha=0.15, color=C_ROUGE)
        gap = tr_mean[-1] - va_mean[-1]
        diag = "Surapprentissage" if gap > 0.1 else \
               "Sous-apprentissage" if va_mean[-1] < 0.6 else "Bon équilibre ✓"
        ax.set_xlabel("Taille de l'échantillon d'entraînement")
        ax.set_ylabel("AUC-ROC")
        ax.set_title(f"Courbe d'apprentissage – Random Forest\n"
                     f"Gap train/val = {gap:.3f} | Diagnostic : {diag}")
        ax.legend()
        ax.set_ylim(0.5, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info(f"**Diagnostic :** Gap = {gap:.3f} → {diag}")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 – PRÉDICTION INDIVIDUELLE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖 Prédiction individuelle":
    st.title("🤖 Prédiction Individuelle du Risque de Violence Physique")
    st.markdown("Renseignez le profil d'une femme pour obtenir sa probabilité estimée.")

    with st.spinner("Chargement du modèle..."):
        pipeline, _ = load_or_train_model(data_path)

    if pipeline is None:
        st.error("Modèle indisponible. Chargez `violence_physique_clean.csv`.")
        st.stop()

    # ── Formulaire ──
    with st.form("prediction_form"):
        st.markdown("#### 👩 Caractéristiques de la femme")
        col1, col2, col3 = st.columns(3)

        with col1:
            age_femme = st.slider("Âge de la femme", 15, 49, 28)
            education_femme = st.selectbox("Éducation (femme)",
                ["Superieur", "Secondaire", "Primaire", "Aucun"], index=1)
            sit_matrimoniale = st.selectbox("Situation matrimoniale",
                ["Mariee", "Union_libre", "Separee", "Divorcee", "Veuve", "Jamais_en_union"])
        with col2:
            enceinte = st.selectbox("Actuellement enceinte", ["Non", "Oui"])
            lecture_journal = st.selectbox("Lecture de journaux",
                ["Jamais", "Moins_1x_sem", "Au_moins_1x_sem", "Presque_chaque_jour"])
            ecoute_radio = st.selectbox("Écoute radio",
                ["Jamais", "Moins_1x_sem", "Au_moins_1x_sem", "Presque_chaque_jour"])
        with col3:
            regarde_tv = st.selectbox("Télévision",
                ["Jamais", "Moins_1x_sem", "Au_moins_1x_sem", "Presque_chaque_jour"])

        st.markdown("#### 👨 Caractéristiques du conjoint")
        col4, col5, col6 = st.columns(3)
        with col4:
            age_conjoint = st.slider("Âge du conjoint", 15, 80, 35)
            education_conjoint = st.selectbox("Éducation (conjoint)",
                ["Superieur", "Secondaire", "Primaire", "Aucun"], index=2)
        with col5:
            conjoint_alcool = st.selectbox("Conjoint consomme de l'alcool", ["Non", "Oui"])
        with col6:
            st.markdown(" ")

        st.markdown("#### 🗺 Contexte géographique")
        col7, col8 = st.columns(2)
        with col7:
            region = st.selectbox("Région", [
                "Centre", "Littoral", "Ouest", "Nord_Ouest", "Sud_Ouest",
                "Adamaoua", "Nord", "Extreme_Nord", "Est", "Sud"])
        with col8:
            milieu = st.selectbox("Milieu de résidence", ["Urbain", "Rural"])

        submitted = st.form_submit_button("🔮 Prédire le risque", use_container_width=True)

    if submitted:
        # Score média calculé
        media_map = {"Jamais": 0, "Moins_1x_sem": 1, "Au_moins_1x_sem": 2, "Presque_chaque_jour": 3}
        score_media = (media_map[lecture_journal] + media_map[ecoute_radio] + media_map[regarde_tv]) / 3
        ecart_age   = age_conjoint - age_femme

        # Construction du vecteur
        profil_dict = {
            "age_femme"          : age_femme,
            "education_femme"    : education_femme,
            "sit_matrimoniale"   : sit_matrimoniale,
            "enceinte"           : enceinte,
            "lecture_journal"    : lecture_journal,
            "ecoute_radio"       : ecoute_radio,
            "regarde_tv"         : regarde_tv,
            "score_media"        : score_media,
            "age_conjoint"       : age_conjoint,
            "ecart_age"          : ecart_age,
            "education_conjoint" : education_conjoint,
            "conjoint_alcool"    : conjoint_alcool,
            "region"             : region,
            "milieu"             : milieu,
        }
        df_profil = pd.DataFrame([profil_dict])
        df_enc    = encode_features(df_profil)
        X_input   = df_enc[FEATURES].values

        prob = pipeline.predict_proba(X_input)[0, 1]

        st.divider()
        col_g, col_r = st.columns([1, 1])

        with col_g:
            fig_gauge = gauge_chart(prob)
            st.pyplot(fig_gauge)
            plt.close()

        with col_r:
            niveau = ("🔴 RISQUE ÉLEVÉ" if prob > 0.55 else
                      "🟠 RISQUE MODÉRÉ" if prob > 0.33 else
                      "🟢 RISQUE FAIBLE")
            st.markdown(f"### {niveau}")
            st.markdown(f"""
| Indicateur | Valeur |
|---|---|
| Probabilité estimée | **{prob*100:.1f}%** |
| Classe prédite | **{'Victime' if prob >= 0.5 else 'Non victime'}** |
| Score médias | **{score_media:.2f} / 3** |
| Écart d'âge | **{ecart_age:+d} ans** |
            """)
            if prob > 0.55:
                st.error("⚠ Ce profil présente des facteurs de risque cumulés. "
                         "Une attention particulière est recommandée.")
            elif prob > 0.33:
                st.warning("Ce profil présente un risque modéré. "
                           "Certains facteurs protecteurs peuvent être renforcés.")
            else:
                st.success("Ce profil présente peu de facteurs de risque identifiés.")

        # Facteurs du profil
        st.divider()
        st.markdown("#### Récapitulatif du profil saisi")
        df_display = pd.DataFrame(profil_dict.items(),
                                  columns=["Variable", "Valeur"])
        df_display["Variable"] = df_display["Variable"].map(
            lambda x: FEATURES_FR.get(x, x))
        st.dataframe(df_display.set_index("Variable"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4 – EXPLICABILITÉ SHAP
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔍 Explicabilité (SHAP)":
    st.title("🔍 Explicabilité du Modèle – Valeurs SHAP")
    st.markdown("Les valeurs SHAP quantifient la contribution de chaque variable "
                "à la probabilité prédite de violence physique.")

    try:
        import shap
    except ImportError:
        st.error("La librairie `shap` n'est pas installée. "
                 "Lancez : `pip install shap`")
        st.stop()

    with st.spinner("Chargement du modèle et calcul des SHAP values..."):
        pipeline, split_data = load_or_train_model(data_path)

    if pipeline is None:
        st.error("Modèle indisponible.")
        st.stop()

    X_test   = split_data["X_test"]
    y_test   = split_data["y_test"]
    model    = pipeline.named_steps["model"]

    # Sous-échantillon pour les calculs SHAP
    N = min(800, X_test.shape[0])
    np.random.seed(42)
    idx   = np.random.choice(X_test.shape[0], N, replace=False)
    X_exp = X_test[idx]
    y_exp = y_test[idx]

    X_exp_df = pd.DataFrame(X_exp, columns=FEATURES)

    @st.cache_data(show_spinner=False)
    def compute_shap(_model, _X_exp):
        explainer = shap.TreeExplainer(_model)
        sv        = explainer.shap_values(_X_exp)
        ev        = explainer.expected_value

        # ── Normalisation sv : liste (multi-classe) → classe 1 ──
        if isinstance(sv, list):
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]

        # ── Normalisation ev : liste / array de longueur quelconque → scalaire ──
        if isinstance(ev, (list, np.ndarray)):
            ev = np.atleast_1d(ev)
            ev = float(ev[1]) if len(ev) > 1 else float(ev[0])
        else:
            ev = float(ev)

        return sv, ev

    sv, ev = compute_shap(model, X_exp)

    # KPIs SHAP
    shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES)
    top_feat = shap_imp.idxmax()
    st.info(f"**Variable la plus influente :** {FEATURES_FR[top_feat]} "
            f"(|SHAP| moyen = {shap_imp[top_feat]:.4f})")

    tabs = st.tabs(["📊 Importance globale", "🐝 Beeswarm",
                    "📉 Dependence plots", "🌊 Profil individuel"])

    # ── Importance globale ──
    with tabs[0]:
        shap_df = pd.DataFrame({
            "Feature"    : [FEATURES_FR[f] for f in FEATURES],
            "SHAP moyen" : shap_imp.values
        }).sort_values("SHAP moyen", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [C_ROUGE if v > shap_imp.mean() else C_GRIS for v in shap_df["SHAP moyen"]]
        ax.barh(shap_df["Feature"], shap_df["SHAP moyen"],
                color=colors, edgecolor="white", height=0.7)
        ax.axvline(shap_imp.mean(), color="gray", ls="--", lw=1, label="Moyenne")
        ax.set_xlabel("|Valeur SHAP| moyenne")
        ax.set_title("Importance globale SHAP – Random Forest\n"
                     "(impact moyen sur la probabilité de violence)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Tableau des importances SHAP**")
        shap_table = pd.DataFrame({
            "Variable (code)" : FEATURES,
            "Variable (label)": [FEATURES_FR[f] for f in FEATURES],
            "|SHAP| moyen"    : shap_imp.values.round(4),
            "Rang"            : shap_imp.rank(ascending=False).astype(int).values
        }).sort_values("|SHAP| moyen", ascending=False)
        st.dataframe(shap_table.set_index("Rang"), use_container_width=True)

# ── Beeswarm ──
with tabs[1]:
    st.markdown("Chaque point représente une observation. "
                "**Rouge** = valeur élevée de la feature | "
                "**Bleu** = valeur faible.")
    
    shap_exp = shap.Explanation(
        values=sv,
        base_values=np.full(len(X_exp_df), ev),
        data=X_exp_df.values,
        feature_names=[FEATURES_FR[f] for f in FEATURES]
    )
    
    # Correction : ne pas passer d'axe, utiliser plot_size pour contrôler la taille
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_exp, max_display=14, show=False)
    plt.title("Beeswarm SHAP – Impact de chaque feature sur la prédiction", fontsize=11)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    # ── Dependence plots ──
    with tabs[2]:
        feat_label = st.selectbox(
            "Choisir la variable",
            options=[FEATURES_FR[f] for f in FEATURES],
            index=[FEATURES_FR[f] for f in FEATURES].index(FEATURES_FR["conjoint_alcool"])
        )
        feat_key = [f for f in FEATURES if FEATURES_FR[f] == feat_label][0]
        feat_idx = FEATURES.index(feat_key)

        fig, ax = plt.subplots(figsize=(9, 5))
        sc = ax.scatter(X_exp_df[feat_key].values, sv[:, feat_idx],
                        c=X_exp_df[feat_key].values,
                        cmap="RdYlGn_r", alpha=0.5, s=12, edgecolors="none")
        ax.axhline(0, color="black", ls="--", lw=0.8)
        ax.set_xlabel(feat_label)
        ax.set_ylabel("Valeur SHAP")
        ax.set_title(f"Dependence plot – {feat_label}\n"
                     "SHAP > 0 : augmente la probabilité de violence")
        plt.colorbar(sc, ax=ax, label=feat_label)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Profil individuel (waterfall) ──
    with tabs[3]:
        y_pred_prob_exp = model.predict_proba(X_exp)[:, 1]
        idx_options     = np.argsort(y_pred_prob_exp)[::-1]

        obs_idx = st.slider("Observation (triées par probabilité décroissante)",
                            0, len(idx_options)-1, 0)
        real_idx = idx_options[obs_idx]
        prob_ind = y_pred_prob_exp[real_idx]
        real_lab = y_exp[real_idx]

        shap_ind = sv[real_idx]
        df_wf = pd.DataFrame({
            "Feature"  : [FEATURES_FR[f] for f in FEATURES],
            "SHAP"     : shap_ind,
            "Valeur"   : X_exp_df.iloc[real_idx].values
        }).sort_values("SHAP")

        top_n = pd.concat([df_wf.head(5), df_wf.tail(5)])

        fig, ax = plt.subplots(figsize=(10, 6))
        colors_wf = [C_ROUGE if v > 0 else C_VERT for v in top_n["SHAP"]]
        labels = [f"{row['Feature']} = {row['Valeur']:.2f}"
                  for _, row in top_n.iterrows()]
        ax.barh(labels, top_n["SHAP"], color=colors_wf,
                edgecolor="white", height=0.65)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Valeur SHAP")
        ax.set_title(
            f"Profil individuel – Probabilité prédite : {prob_ind:.3f} "
            f"| Classe réelle : {'Victime' if real_lab == 1 else 'Non victime'}",
            fontsize=10
        )
        rouge_p = mpatches.Patch(color=C_ROUGE, label="↑ Augmente le risque")
        vert_p  = mpatches.Patch(color=C_VERT,  label="↓ Diminue le risque")
        ax.legend(handles=[rouge_p, vert_p])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col_m, col_n = st.columns(2)
        col_m.metric("Probabilité prédite", f"{prob_ind*100:.1f}%")
        col_n.metric("Classe réelle", "Victime" if real_lab == 1 else "Non victime")


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "📚 **Source :** Enquête Démographique et de Santé – Cameroun 2018 (EDS-V, CMIR71FL.dta)  |  "
    "🤖 **Modèle :** Random Forest (n_estimators=200, max_depth=8, class_weight='balanced')  |  "
    "📊 **AUC-ROC :** 0.7425  |  **F1-score :** 0.5919  |  **Prévalence :** 33.3%"
)
