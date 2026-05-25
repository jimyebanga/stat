"""
streamlit_app.py — Violence physique EDS Cameroun 2018
Source : cameroun_violence_dataset.csv (version allégée de CMIR71FL)

COLONNES DU CSV :
  V744A–V744E          : formes de violence (binaires)
  education_femme      : 0–3
  age_femme            : float
  exposition_medias    : 0/1
  nb_enfants           : entier
  quintile_richesse    : 1–5
  femme_travaille      : 0/1
  education_conjoint   : 0–2
  alcool_conjoint      : 0/1
  milieu_rural         : 0/1
  television           : 0–2
  radio                : 0–2
  violence_physique    : 0/1  (variable cible)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    accuracy_score, roc_curve, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES — mapping des modalités pour l'affichage et la prédiction
# ══════════════════════════════════════════════════════════════════════════════

CAT_COLS = [
    "education_femme", "education_conjoint", "milieu_rural",
    "exposition_medias", "alcool_conjoint", "quintile_richesse",
    "femme_travaille", "television", "radio"
]
NUM_COLS = ["age_femme", "nb_enfants"]

LABELS = {
    "education_femme":    {0:"Aucun", 1:"Primaire", 2:"Secondaire", 3:"Supérieur"},
    "education_conjoint": {0:"Aucun", 1:"Primaire", 2:"Sec./Sup."},
    "milieu_rural":       {0:"Urbain", 1:"Rural"},
    "exposition_medias":  {0:"Non", 1:"Oui"},
    "alcool_conjoint":    {0:"Non", 1:"Oui"},
    "quintile_richesse":  {1:"Q1-Pauvre", 2:"Q2", 3:"Q3-Moyen", 4:"Q4", 5:"Q5-Riche"},
    "femme_travaille":    {0:"Non", 1:"Oui"},
    "television":         {0:"Jamais", 1:"Parfois", 2:"Souvent"},
    "radio":              {0:"Jamais", 1:"Parfois", 2:"Souvent"},
}

FORMES_VIOLENCE = {
    "V744A": "Poussée / secouée",
    "V744B": "Gifle",
    "V744C": "Bras tordu / cheveux tirés",
    "V744D": "Coup de poing / objet",
    "V744E": "Brûlure / ébouillantée",
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CHARGEMENT & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def charger_et_preparer(chemin: str):
    """
    Lit le CSV, fait le get_dummies, split train/test.
    Retourne tout ce dont l'app a besoin, y compris la liste exacte
    des colonnes d'entraînement (pour l'alignement à la prédiction).
    """
    df = pd.read_csv(chemin)

    # --- encodage one-hot (drop_first=True = même logique que le notebook) ---
    df_enc = pd.get_dummies(df[CAT_COLS], columns=CAT_COLS, drop_first=True)
    for col in NUM_COLS:
        df_enc[col] = df[col].values

    X = df_enc.astype(float)
    y = df["violence_physique"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X, y, X_train, X_test, y_train, y_test, df, list(X.columns)


def seuil_optimal(y_prob, y_true):
    """Seuil qui maximise le F1-macro (même logique que le notebook)."""
    seuils = np.arange(0.25, 0.65, 0.005)
    scores = [
        f1_score(y_true, (y_prob >= s).astype(int), average="macro")
        for s in seuils
    ]
    return seuils[np.argmax(scores)]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ENTRAÎNEMENT (cache_resource = objets réutilisés entre reruns)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def entrainer_modeles(_X_train, _y_train, _X_test, _y_test):
    """
    Entraîne les 4 modèles du notebook avec leurs hyperparamètres exacts.
    L'underscore devant les arrays numpy contourne la limitation de hash
    de st.cache_resource.
    """
    X_train, y_train = _X_train, _y_train
    X_test,  y_test  = _X_test,  _y_test
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    # 1. Régression Logistique
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced", C=1.0,
            max_iter=2000, solver="lbfgs", random_state=42
        ))
    ])
    pipe_lr.fit(X_train, y_train)

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=30,
        min_samples_split=60, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 3. XGBoost
    xgb_m = xgb.XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=15,
        gamma=0.5, reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=ratio, eval_metric="auc",
        random_state=42, n_jobs=-1
    )
    xgb_m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # 4. Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=30, random_state=42
    )
    sample_w = np.where(y_train == 1, ratio, 1.0)
    gb.fit(X_train, y_train, sample_weight=sample_w)

    modeles = {
        "Logistique":        pipe_lr,
        "Random Forest":     rf,
        "XGBoost":           xgb_m,
        "Gradient Boosting": gb,
    }

    resultats = {}
    for name, model in modeles.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        seuil   = seuil_optimal(y_proba, y_test)
        y_pred  = (y_proba >= seuil).astype(int)
        resultats[name] = {
            "model":     model,
            "seuil":     round(float(seuil), 3),
            "y_proba":   y_proba,
            "y_pred":    y_pred,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 3),
            "AUC-ROC":   round(roc_auc_score(y_test, y_proba), 3),
            "Rappel":    round(recall_score(y_test, y_pred), 3),
            "Précision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "F1 Macro":  round(f1_score(y_test, y_pred, average="macro"), 3),
        }

    return modeles, resultats


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Violence physique — EDS Cameroun 2018",
    page_icon="📊", layout="wide"
)
st.title("📊 Violence physique faite aux femmes en couple")
st.caption("EDS Cameroun 2018 — Analyse ML | EBANGA MBALLA")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
chemin = st.sidebar.text_input(
    "Fichier CSV",
    value="cameroun_violence_dataset.csv",
    help="Chemin vers cameroun_violence_dataset.csv"

)

# Chargement automatique au démarrage
try:
    with st.spinner("Chargement des données..."):
        X, y, X_train, X_test, y_train, y_test, df_raw, colonnes = \
            charger_et_preparer(chemin)

    with st.spinner("Entraînement des modèles ML (première fois ~1 min)..."):
        modeles, resultats = entrainer_modeles(X_train, y_train, X_test, y_test)

    st.sidebar.success(f"✅ {len(X):,} observations chargées")
    st.sidebar.info(
        f"Violence : {y.sum():,} cas ({y.mean()*100:.1f}%)\n\n"
        f"Train : {len(X_train):,} | Test : {len(X_test):,}"
    )

except FileNotFoundError:
    st.error(
        f"❌ Fichier introuvable : `{chemin}`\n\n"
        "Placez `cameroun_violence_dataset.csv` dans le même dossier "
        "que ce script, ou corrigez le chemin ci-dessus."
    )
    st.stop()

# ── Onglets ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Prévalence & descriptif",
    "🤖 Comparaison des modèles",
    "🌲 Importance des variables",
    "🔮 Prédiction individuelle",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Prévalence
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Prévalence de la violence physique")

    n_total = len(df_raw)
    n_viol  = int(df_raw["violence_physique"].sum())
    prev    = n_viol / n_total * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Effectif total", f"{n_total:,}")
    c2.metric("Cas de violence", f"{n_viol:,}")
    c3.metric("Prévalence globale", f"{prev:.1f}%")

    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            [n_total - n_viol, n_viol],
            labels=[f"Pas de violence\n{100-prev:.1f}%", f"Violence\n{prev:.1f}%"],
            colors=["#2ecc71", "#c0392b"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2.5)
        )
        ax.set_title("Prévalence globale", fontweight="bold")
        st.pyplot(fig)

    with col_bar:
        vals  = [(df_raw[c] == 1).mean() * 100 for c in FORMES_VIOLENCE]
        labs  = list(FORMES_VIOLENCE.values())
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(labs, vals,
                       color=["#c0392b","#e74c3c","#e67e22","#d35400","#922b21"],
                       edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("Prévalence (%)")
        ax.set_title("Par forme de violence", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    # Prévalence par variable catégorielle
    st.subheader("Prévalence selon les facteurs")
    var_choisie = st.selectbox(
        "Variable à explorer",
        options=list(LABELS.keys()),
        format_func=lambda x: x.replace("_", " ").title()
    )
    prev_var = (
        df_raw.groupby(var_choisie)["violence_physique"]
        .mean().mul(100).round(1).reset_index()
    )
    prev_var.columns = ["Code", "Prévalence (%)"]
    prev_var["Modalité"] = prev_var["Code"].map(LABELS[var_choisie])
    prev_glob = prev

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors_v = ["#c0392b" if v > prev_glob else "#2980b9"
                for v in prev_var["Prévalence (%)"]]
    ax.barh(prev_var["Modalité"], prev_var["Prévalence (%)"],
            color=colors_v, edgecolor="white")
    ax.axvline(prev_glob, color="#e67e22", linestyle="--", lw=1.5,
               label=f"Moyenne {prev_glob:.1f}%")
    for _, row in prev_var.iterrows():
        ax.text(row["Prévalence (%)"] + 0.3,
                list(prev_var["Modalité"]).index(row["Modalité"]),
                f"{row['Prévalence (%)']:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Prévalence (%)")
    ax.set_title(f"Prévalence par {var_choisie.replace('_',' ')}", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Comparaison des modèles
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Performances comparées")

    rows = []
    for name, res in resultats.items():
        rows.append({
            "Modèle":    name,
            "Seuil":     res["seuil"],
            "Accuracy":  res["Accuracy"],
            "AUC-ROC":   res["AUC-ROC"],
            "Rappel":    res["Rappel"],
            "Précision": res["Précision"],
            "F1 Macro":  res["F1 Macro"],
        })
    df_perf = pd.DataFrame(rows)
    st.dataframe(
        df_perf.set_index("Modèle").style.highlight_max(
            axis=0, color="#d5f5e3",
            subset=["AUC-ROC", "Rappel", "F1 Macro"]
        ),
        use_container_width=True
    )

    col_roc, col_pr = st.columns(2)
    colors_ = ["#1a1a2e", "#c0392b", "#e67e22", "#27ae60"]
    styles_ = ["-", "--", "-.", "dotted"]

    with col_roc:
        fig, ax = plt.subplots(figsize=(6, 5))
        for (name, res), color, ls in zip(resultats.items(), colors_, styles_):
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
                    label=f"{name} (AUC={res['AUC-ROC']})")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("Faux positifs"); ax.set_ylabel("Sensibilité")
        ax.set_title("Courbes ROC", fontweight="bold")
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    with col_pr:
        fig, ax = plt.subplots(figsize=(6, 5))
        for (name, res), color, ls in zip(resultats.items(), colors_, styles_):
            prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
            ap = average_precision_score(y_test, res["y_proba"])
            ax.plot(rec, prec, color=color, lw=2, linestyle=ls,
                    label=f"{name} (AP={ap:.3f})")
        ax.axhline(y_test.mean(), color="gray", linestyle=":", lw=1,
                   label=f"Baseline ({y_test.mean():.2f})")
        ax.set_xlabel("Rappel"); ax.set_ylabel("Précision")
        ax.set_title("Courbes Précision-Rappel", fontweight="bold")
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    st.subheader("Matrices de confusion")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, res) in zip(axes, resultats.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, res["y_pred"],
            display_labels=["Non", "Oui"],
            cmap="Reds", ax=ax, colorbar=False
        )
        ax.set_title(name, fontweight="bold", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Importance des variables
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Importance des variables")

    modele_choisi = st.selectbox(
        "Choisir le modèle",
        ["XGBoost", "Random Forest", "Gradient Boosting"]
    )
    mod_obj = modeles[modele_choisi]
    imp = pd.Series(
        mod_obj.feature_importances_, index=X_train.columns
    ).sort_values(ascending=True).tail(15)
    med = imp.median()
    col_i = ["#c0392b" if v >= med else "#2980b9" for v in imp.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp.index, imp.values, color=col_i, edgecolor="white")
    ax.axvline(med, color="#e67e22", linestyle="--", lw=1.5, label="Médiane")
    rouge = mpatches.Patch(color="#c0392b", label="Au-dessus médiane")
    bleu  = mpatches.Patch(color="#2980b9", label="En-dessous médiane")
    ax.legend(handles=[rouge, bleu, plt.Line2D([0], [0], color="#e67e22",
              linestyle="--", label="Médiane")], fontsize=9)
    ax.set_title(f"Top 15 variables — {modele_choisi}", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Prédiction individuelle
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔮 Prédiction pour un profil individuel")
    st.info("Renseignez le profil d'une femme pour estimer la probabilité "
            "de violence physique selon chaque modèle.")

    c1, c2, c3 = st.columns(3)
    with c1:
        p_educ_f = st.selectbox("Éducation de la femme",
            options=list(LABELS["education_femme"].keys()),
            format_func=lambda x: LABELS["education_femme"][x])
        p_educ_c = st.selectbox("Éducation du conjoint",
            options=list(LABELS["education_conjoint"].keys()),
            format_func=lambda x: LABELS["education_conjoint"][x])
        p_milieu = st.selectbox("Milieu de résidence",
            options=list(LABELS["milieu_rural"].keys()),
            format_func=lambda x: LABELS["milieu_rural"][x])
    with c2:
        p_medias  = st.selectbox("Exposition aux médias",
            options=list(LABELS["exposition_medias"].keys()),
            format_func=lambda x: LABELS["exposition_medias"][x])
        p_alcool  = st.selectbox("Alcool du conjoint",
            options=list(LABELS["alcool_conjoint"].keys()),
            format_func=lambda x: LABELS["alcool_conjoint"][x])
        p_richesse = st.selectbox("Quintile de richesse",
            options=list(LABELS["quintile_richesse"].keys()),
            format_func=lambda x: LABELS["quintile_richesse"][x])
    with c3:
        p_travail = st.selectbox("Femme travaille",
            options=list(LABELS["femme_travaille"].keys()),
            format_func=lambda x: LABELS["femme_travaille"][x])
        p_tv    = st.selectbox("Télévision",
            options=list(LABELS["television"].keys()),
            format_func=lambda x: LABELS["television"][x])
        p_radio = st.selectbox("Radio",
            options=list(LABELS["radio"].keys()),
            format_func=lambda x: LABELS["radio"][x])

    p_age        = st.slider("Âge", 15, 49, 28)
    p_nb_enfants = st.slider("Nombre d'enfants", 0, 15, 2)

    if st.button("📊 Estimer la probabilité"):

        # ── Reconstruction du profil avec EXACTEMENT les mêmes colonnes ──────
        profil_brut = pd.DataFrame([{
            "education_femme":    p_educ_f,
            "education_conjoint": p_educ_c,
            "milieu_rural":       p_milieu,
            "exposition_medias":  p_medias,
            "alcool_conjoint":    p_alcool,
            "quintile_richesse":  p_richesse,
            "femme_travaille":    p_travail,
            "television":         p_tv,
            "radio":              p_radio,
        }])

        profil_enc = pd.get_dummies(profil_brut[CAT_COLS],
                                    columns=CAT_COLS, drop_first=True)
        profil_enc["age_femme"]   = float(p_age)
        profil_enc["nb_enfants"]  = float(p_nb_enfants)

        # Alignement critique : mêmes colonnes que l'entraînement
        profil_final = profil_enc.reindex(columns=colonnes, fill_value=0).astype(float)

        st.subheader("Résultats")
        res_cols = st.columns(len(modeles))
        for col_ui, (name, model) in zip(res_cols, modeles.items()):
            proba = model.predict_proba(profil_final)[0, 1]
            seuil = resultats[name]["seuil"]
            pred  = "🔴 Violence" if proba >= seuil else "🟢 Pas de violence"
            col_ui.metric(label=name, value=f"{proba*100:.1f}%", delta=pred)

        # Jauge visuelle de consensus
        probas  = [modeles[n].predict_proba(profil_final)[0, 1] for n in modeles]
        moy_p   = np.mean(probas)
        st.markdown("---")
        st.markdown(f"**Probabilité moyenne (4 modèles) : {moy_p*100:.1f}%**")
        couleur = "#c0392b" if moy_p >= 0.4 else "#e67e22" if moy_p >= 0.25 else "#27ae60"
        st.markdown(
            f"<div style='background:{couleur};border-radius:8px;"
            f"height:22px;width:{min(moy_p*100,100):.0f}%'></div>",
            unsafe_allow_html=True
        )
