"""
streamlit_app.py — Violence physique EDS Cameroun 2018
Intégration correcte du ML du notebook EBANGA_MBALLA.ipynb

STRUCTURE :
  1. Chargement & preprocessing (mis en cache)
  2. Entraînement des modèles (mis en cache)
  3. Interface utilisateur : tableaux, graphiques, prédiction individuelle
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Imports ML ────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    accuracy_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import statsmodels.api as sm

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PREPROCESSING (identique au notebook, mis en cache)
# C'est ici que se règle l'incohérence : UNE SEULE version du preprocessing
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def charger_et_preparer(chemin_fichier: str):
    """
    Charge CMIR71FL.dta et reproduit EXACTEMENT le preprocessing du notebook.
    Le cache Streamlit garantit que c'est fait une seule fois.
    Retourne : df_enc (X), y, X_train, X_test, y_train, y_test, dc, colonnes
    """
    df = pd.read_stata(chemin_fichier, convert_categoricals=False)

    # Variable dépendante
    cols_v = ["V744A", "V744B", "V744C", "V744D", "V744E"]
    for c in cols_v:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["violence_physique"] = (df[cols_v] == 1).any(axis=1).astype(int)
    df["statut_marital"] = pd.to_numeric(df["V501"], errors="coerce")
    dc = df[df["statut_marital"].isin([1, 2])].copy()

    # Recodages numériques (mêmes que le dernier bloc du notebook)
    for col, src in [
        ("educ_femme", "V106"), ("educ_conj", "V701"), ("milieu", "V025"),
        ("medias", "V751"), ("alcool", "V745A"), ("richesse", "V190"),
        ("travail", "V714"), ("tv", "V159"), ("radio", "V158"),
        ("age", "V012"), ("nb_enfants", "V201")
    ]:
        dc[col] = pd.to_numeric(dc[src], errors="coerce")
    dc["educ_conj"] = dc["educ_conj"].replace(8, np.nan)
    dc["travail"]   = dc["travail"].replace(8, np.nan)

    cat_cols = ["educ_femme", "educ_conj", "milieu", "medias", "alcool",
                "richesse", "travail", "tv", "radio"]

    df_model = dc[cat_cols + ["age", "nb_enfants", "violence_physique"]].dropna()
    df_enc   = pd.get_dummies(df_model[cat_cols], columns=cat_cols, drop_first=True)
    df_enc["age"]        = df_model["age"].values
    df_enc["nb_enfants"] = df_model["nb_enfants"].values

    X = df_enc.astype(float)
    y = df_model["violence_physique"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # IMPORTANT : on retourne aussi les noms de colonnes
    # pour pouvoir reconstruire le vecteur de prédiction individuelle
    colonnes = list(X.columns)

    return X, y, X_train, X_test, y_train, y_test, dc, colonnes, df_enc


def seuil_optimal(y_prob, y_true):
    """Trouve le seuil qui maximise le F1-macro (même logique que le notebook)."""
    seuils = np.arange(0.25, 0.65, 0.005)
    scores = [f1_score(y_true, (y_prob >= s).astype(int), average="macro")
              for s in seuils]
    best = seuils[np.argmax(scores)]
    return best


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ENTRAÎNEMENT (mis en cache, fait une seule fois)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def entrainer_modeles(X_train, y_train, X_test, y_test):
    """
    Entraîne les 4 modèles identiquement au notebook.
    cache_resource = les objets Python (modèles) sont réutilisés entre reruns.
    """
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
        "Logistique":       pipe_lr,
        "Random Forest":    rf,
        "XGBoost":          xgb_m,
        "Gradient Boosting": gb,
    }

    # Calcul des métriques avec seuil optimal (même logique que notebook)
    resultats = {}
    for name, model in modeles.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        seuil   = seuil_optimal(y_proba, y_test)
        y_pred  = (y_proba >= seuil).astype(int)
        resultats[name] = {
            "model":    model,
            "seuil":    round(seuil, 3),
            "y_proba":  y_proba,
            "y_pred":   y_pred,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "AUC-ROC":  round(roc_auc_score(y_test, y_proba), 3),
            "Rappel":   round(recall_score(y_test, y_pred), 3),
            "Précision": round(precision_score(y_test, y_pred), 3),
            "F1-score": round(f1_score(y_test, y_pred, average="macro"), 3),
        }

    return modeles, resultats


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Violence physique — EDS Cameroun 2018",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Violence physique faite aux femmes en couple")
st.caption("EDS Cameroun 2018 — Analyse ML | EBANGA MBALLA")

# ── Chargement du fichier ─────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
chemin = st.sidebar.text_input(
    "Chemin vers CMIR71FL.dta",
    value="CMIR71FL.dta",
    help="Fichier Stata EDS Cameroun 2018"
)

if st.sidebar.button("🚀 Charger et entraîner les modèles") or True:
    try:
        with st.spinner("Chargement des données et preprocessing..."):
            X, y, X_train, X_test, y_train, y_test, dc, colonnes, df_enc = \
                charger_et_preparer(chemin)

        with st.spinner("Entraînement des modèles ML (peut prendre ~1 min)..."):
            modeles, resultats = entrainer_modeles(
                X_train, y_train, X_test, y_test
            )

        st.sidebar.success(f"✅ Données chargées : {len(X):,} observations")

    except FileNotFoundError:
        st.error(f"❌ Fichier introuvable : `{chemin}`\n\n"
                 "Placez `CMIR71FL.dta` dans le même dossier que ce script "
                 "ou corrigez le chemin dans la barre latérale.")
        st.stop()

    # ── Onglets ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Prévalence", "🤖 Comparaison modèles",
        "🌲 Importance variables", "🔮 Prédiction individuelle"
    ])

    # ── TAB 1 : Prévalence ────────────────────────────────────────────────────
    with tab1:
        st.subheader("Prévalence de la violence physique")
        n_total = len(dc)
        n_viol  = int(dc["violence_physique"].sum())
        prev    = n_viol / n_total * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Effectif total", f"{n_total:,}")
        col2.metric("Cas de violence", f"{n_viol:,}")
        col3.metric("Prévalence globale", f"{prev:.1f}%")

        # Graphique camembert
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            [n_total - n_viol, n_viol],
            labels=[f"Pas de violence\n{100-prev:.1f}%", f"Violence\n{prev:.1f}%"],
            colors=["#2ecc71", "#c0392b"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2)
        )
        ax.set_title("Prévalence", fontweight="bold")
        st.pyplot(fig, use_container_width=False)

    # ── TAB 2 : Comparaison modèles ───────────────────────────────────────────
    with tab2:
        st.subheader("Comparaison des performances")

        rows = []
        for name, res in resultats.items():
            rows.append({
                "Modèle":    name,
                "Seuil":     res["seuil"],
                "Accuracy":  res["Accuracy"],
                "AUC-ROC":   res["AUC-ROC"],
                "Rappel":    res["Rappel"],
                "Précision": res["Précision"],
                "F1 Macro":  res["F1-score"],
            })
        df_perf = pd.DataFrame(rows)
        st.dataframe(df_perf.set_index("Modèle"), use_container_width=True)

        # Courbes ROC
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_ = ["#1a1a2e", "#c0392b", "#e67e22", "#27ae60"]
        styles_ = ["-", "--", "-.", "dotted"]
        for (name, res), color, ls in zip(resultats.items(), colors_, styles_):
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
                    label=f"{name} (AUC={res['AUC-ROC']})")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Sensibilité")
        ax.set_title("Courbes ROC", fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

        # Matrices de confusion
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

    # ── TAB 3 : Importance variables ──────────────────────────────────────────
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
        ax.set_title(f"Top 15 variables — {modele_choisi}", fontweight="bold")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

    # ── TAB 4 : Prédiction individuelle ──────────────────────────────────────
    with tab4:
        st.subheader("🔮 Prédiction pour un profil individuel")
        st.info("Saisissez les caractéristiques d'une femme pour estimer "
                "la probabilité de violence physique.")

        col1, col2, col3 = st.columns(3)
        with col1:
            educ_femme = st.selectbox("Éducation de la femme",
                [0, 1, 2, 3],
                format_func=lambda x: {0:"Aucun",1:"Primaire",
                                        2:"Secondaire",3:"Supérieur"}[x])
            educ_conj = st.selectbox("Éducation du conjoint",
                [0, 1, 2, 3],
                format_func=lambda x: {0:"Aucun",1:"Primaire",
                                        2:"Secondaire",3:"Supérieur"}[x])
            milieu = st.selectbox("Milieu", [1, 2],
                format_func=lambda x: {1:"Urbain", 2:"Rural"}[x])
        with col2:
            medias = st.selectbox("Exposition aux médias", [0, 1],
                format_func=lambda x: {0:"Non", 1:"Oui"}[x])
            alcool = st.selectbox("Consommation alcool conjoint", [0, 1],
                format_func=lambda x: {0:"Non", 1:"Oui"}[x])
            richesse = st.selectbox("Quintile de richesse",
                [1, 2, 3, 4, 5],
                format_func=lambda x: {1:"Q1-Pauvre",2:"Q2",3:"Q3-Moyen",
                                        4:"Q4",5:"Q5-Riche"}[x])
        with col3:
            travail = st.selectbox("Femme travaille", [0, 1],
                format_func=lambda x: {0:"Non", 1:"Oui"}[x])
            tv = st.selectbox("TV", [0, 1, 2, 3],
                format_func=lambda x: {0:"Jamais",1:"Parfois",
                                        2:"Souvent",3:"Presque chaque jour"}[x])
            radio = st.selectbox("Radio", [0, 1, 2, 3],
                format_func=lambda x: {0:"Jamais",1:"Parfois",
                                        2:"Souvent",3:"Presque chaque jour"}[x])
        age = st.slider("Âge", 15, 49, 28)
        nb_enfants = st.slider("Nombre d'enfants", 0, 15, 2)

        if st.button("📊 Calculer la probabilité"):
            # Reconstruction du vecteur de features AVEC LES MÊMES COLONNES
            # que celles issues du get_dummies d'entraînement
            profil_brut = pd.DataFrame([{
                "educ_femme": educ_femme, "educ_conj": educ_conj,
                "milieu": milieu, "medias": medias, "alcool": alcool,
                "richesse": richesse, "travail": travail,
                "tv": tv, "radio": radio,
                "age": float(age), "nb_enfants": float(nb_enfants)
            }])

            cat_cols = ["educ_femme", "educ_conj", "milieu", "medias",
                        "alcool", "richesse", "travail", "tv", "radio"]
            profil_enc = pd.get_dummies(profil_brut[cat_cols],
                                         columns=cat_cols, drop_first=True)
            profil_enc["age"]        = profil_brut["age"].values
            profil_enc["nb_enfants"] = profil_brut["nb_enfants"].values

            # ALIGNEMENT DES COLONNES (résout l'incohérence principale)
            profil_final = profil_enc.reindex(columns=colonnes, fill_value=0).astype(float)

            st.subheader("Résultats par modèle")
            res_cols = st.columns(len(modeles))
            for col_ui, (name, model) in zip(res_cols, modeles.items()):
                proba = model.predict_proba(profil_final)[0, 1]
                seuil = resultats[name]["seuil"]
                pred  = "🔴 Violence" if proba >= seuil else "🟢 Pas de violence"
                col_ui.metric(
                    label=name,
                    value=f"{proba*100:.1f}%",
                    delta=pred
                )
