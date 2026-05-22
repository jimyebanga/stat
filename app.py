# =============================================================
# APPLICATION STREAMLIT — Violence Domestique au Cameroun
# EDS-MICS 2018 | Analyse ML
# Lancer : streamlit run app.py
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    f1_score, classification_report, roc_curve,
    ConfusionMatrixDisplay
)
import xgboost as xgb

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Violence physique Domestique — Cameroun EDS 2018",
    page_icon="🇨🇲",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #1a1a2e !important; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 12px; padding: 20px; color: white;
    text-align: center; border-left: 4px solid #c0392b; margin-bottom: 10px;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #e74c3c; }
.metric-card .lbl { font-size: 0.78rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
.section-header {
    background: linear-gradient(90deg, #c0392b, #e74c3c);
    color: white; padding: 10px 18px; border-radius: 8px;
    font-size: 1rem; font-weight: 600; margin: 16px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Fonctions ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    import pyreadstat, tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dta") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    df_raw, _ = pyreadstat.read_dta(tmp_path)
    os.unlink(tmp_path)

    variables = {
        "V744A": "viol_pousse",   "V744B": "viol_gifle",
        "V744C": "viol_bras",     "V744D": "viol_coup",
        "V744E": "viol_brulure",  "V106":  "educ_femme",
        "V701":  "educ_conjoint", "V012":  "age_femme",
        "V751":  "expo_medias",   "V745A": "alcool_conjoint",
        "V025":  "milieu",        "V501":  "statut_marital",
    }
    cols = [c for c in variables if c in df_raw.columns]
    df = df_raw[cols].copy().rename(columns={k: v for k, v in variables.items() if k in cols})

    # Variable dépendante
    cols_v = [c for c in ["viol_pousse","viol_gifle","viol_bras","viol_coup","viol_brulure"] if c in df.columns]
    for c in cols_v:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["violence_physique"] = (df[cols_v] == 1).any(axis=1).astype(int)

    # Recodages
    df["educ_femme"]     = pd.to_numeric(df.get("educ_femme",     pd.Series(dtype=float)), errors="coerce").map({0:"Aucun",1:"Primaire",2:"Secondaire",3:"Superieur"})
    df["educ_conjoint"]  = pd.to_numeric(df.get("educ_conjoint",  pd.Series(dtype=float)), errors="coerce").map({0:"Aucun",1:"Primaire",2:"Sec_Sup",3:"Sec_Sup",8:np.nan})
    df["milieu"]         = pd.to_numeric(df.get("milieu",         pd.Series(dtype=float)), errors="coerce").map({1:"Urbain",2:"Rural"})
    df["expo_medias"]    = pd.to_numeric(df.get("expo_medias",    pd.Series(dtype=float)), errors="coerce").map({0:"Non",1:"Oui"})
    df["alcool_conjoint"]= pd.to_numeric(df.get("alcool_conjoint",pd.Series(dtype=float)), errors="coerce").map({0:"Non",1:"Oui"})
    df["age_femme"]      = pd.to_numeric(df.get("age_femme",      pd.Series(dtype=float)), errors="coerce")
    df["statut_marital"] = pd.to_numeric(df.get("statut_marital", pd.Series(dtype=float)), errors="coerce")

    if "statut_marital" in df.columns:
        df = df[df["statut_marital"].isin([1, 2])]

    features = [f for f in ["educ_femme","educ_conjoint","age_femme","expo_medias","alcool_conjoint","milieu"] if f in df.columns]
    df_clean = df[features + ["violence_physique"]].dropna()
    return df_clean, features


@st.cache_data(show_spinner=False)
def train_models(_df):
    cat_cols = [c for c in ["educ_femme","educ_conjoint","expo_medias","alcool_conjoint","milieu"] if c in _df.columns]
    df_enc = pd.get_dummies(_df, columns=cat_cols, drop_first=False)
    X = df_enc.drop(columns=["violence_physique"])
    y = df_enc["violence_physique"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Régression Logistique": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest":         RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost":               xgb.XGBClassifier(n_estimators=200, scale_pos_weight=ratio, eval_metric="logloss", random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "AUC-ROC":  round(roc_auc_score(y_test, y_proba), 3),
            "Rappel":   round(recall_score(y_test, y_pred), 3),
            "F1-score": round(f1_score(y_test, y_pred), 3),
            "model": model, "y_proba": y_proba, "y_pred": y_pred,
            "report": classification_report(y_test, y_pred, target_names=["Pas de violence","Violence"], output_dict=True),
        }
    return results, X_train, X_test, y_train, y_test


LABELS = {
    "educ_femme":      "Éducation de la femme",
    "educ_conjoint":   "Éducation du conjoint",
    "milieu":          "Milieu de résidence",
    "expo_medias":     "Exposition aux médias",
    "alcool_conjoint": "Alcool du conjoint",
}


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇨🇲 EDS Cameroun 2018")
    st.markdown("Analyse de la **violence physique domestique** chez les femmes de 15–49 ans.")
    st.divider()
    uploaded = st.file_uploader("📂 Charger CMIR71FL.dta", type=["dta"])
    st.divider()
    st.markdown("**Variables utilisées**")
    for k, v in LABELS.items():
        st.markdown(f"- {v}")
    st.markdown("- Âge de la femme")
    st.divider()
    st.caption("Modèles : Logistique · Random Forest · XGBoost")


# ── HEADER ────────────────────────────────────────────────────
st.title("Violence Domestique au Cameroun")
st.subheader("Analyse Statistique & Machine Learning — EDS-MICS 2018")
st.divider()

if uploaded is None:
    st.info("⬅️  Chargez le fichier **CMIR71FL.dta** dans le panneau gauche pour démarrer.")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="metric-card"><div class="val">29,3 %</div><div class="lbl">Prévalence nationale</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><div class="val">8 060</div><div class="lbl">Femmes analysées</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="metric-card"><div class="val">3</div><div class="lbl">Modèles ML comparés</div></div>', unsafe_allow_html=True)
    st.stop()


# ── CHARGEMENT & ENTRAÎNEMENT ─────────────────────────────────
with st.spinner("Chargement des données..."):
    try:
        df_clean, features = load_data(uploaded)
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        st.stop()

with st.spinner("Entraînement des modèles (30–60 s)..."):
    results, X_train, X_test, y_train, y_test = train_models(df_clean)

prevalence = df_clean["violence_physique"].mean() * 100
n_total    = len(df_clean)
n_pos      = int(df_clean["violence_physique"].sum())
best       = max(results, key=lambda k: results[k]["AUC-ROC"])

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="metric-card"><div class="val">{prevalence:.1f} %</div><div class="lbl">Prévalence</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="metric-card"><div class="val">{n_total:,}</div><div class="lbl">Échantillon</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="metric-card"><div class="val">{n_pos:,}</div><div class="lbl">Cas de violence</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="metric-card"><div class="val">{results[best]["AUC-ROC"]}</div><div class="lbl">Meilleur AUC ({best.split()[0]})</div></div>', unsafe_allow_html=True)

st.divider()

# ── ONGLETS ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Statistiques descriptives",
    "🔬 Analyse bivariée",
    "🤖 Modèles ML",
    "📈 Courbes ROC",
    "🔍 Importance des variables",
])


# ══ TAB 1 — DESCRIPTIF ══════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Distribution de l\'âge</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df_clean["age_femme"], bins=20, color="#c0392b", edgecolor="white", alpha=0.85)
        ax.axvline(df_clean["age_femme"].mean(), color="#1a1a2e", linestyle="--", linewidth=2,
                   label=f"Moy. = {df_clean['age_femme'].mean():.1f} ans")
        ax.set_xlabel("Âge (années)"); ax.set_ylabel("Effectif")
        ax.set_title("Distribution de l'âge des femmes", fontweight="bold")
        ax.legend(); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        vals_pie = [n_total - n_pos, n_pos]
        ax.pie(vals_pie, labels=["Pas de violence\n(70,7%)", "Violence\n(29,3%)"],
               colors=["#2ecc71","#c0392b"], autopct="%1.1f%%", startangle=90,
               wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("Prévalence de la violence physique", fontweight="bold")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Répartition par variables catégorielles</div>', unsafe_allow_html=True)
    cat_vars = [c for c in ["educ_femme","educ_conjoint","milieu","expo_medias","alcool_conjoint"] if c in df_clean.columns]
    cols = st.columns(len(cat_vars))
    palette = ["#1a1a2e","#c0392b","#e67e22","#27ae60","#8e44ad"]

    for i, var in enumerate(cat_vars):
        with cols[i]:
            counts = df_clean[var].value_counts()
            fig, ax = plt.subplots(figsize=(3, 2.8))
            ax.bar(counts.index, counts.values, color=palette[:len(counts)], edgecolor="white")
            ax.set_title(LABELS.get(var, var), fontsize=9, fontweight="bold")
            ax.tick_params(axis="x", labelsize=7, rotation=25)
            ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Tableau de fréquences</div>', unsafe_allow_html=True)
    rows = []
    for var in cat_vars:
        for cat, cnt in df_clean[var].value_counts().items():
            rows.append({"Variable": LABELS.get(var, var), "Modalité": cat,
                         "Effectif": cnt, "% ": f"{cnt/n_total*100:.1f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══ TAB 2 — BIVARIÉE ════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Prévalence de la violence selon chaque facteur</div>', unsafe_allow_html=True)
    from scipy.stats import chi2_contingency

    for var in cat_vars:
        prev = df_clean.groupby(var)["violence_physique"].mean().mul(100).round(1).reset_index()
        prev.columns = ["Modalité", "Prévalence (%)"]
        ct = pd.crosstab(df_clean[var], df_clean["violence_physique"])
        chi2, p, dof, _ = chi2_contingency(ct)

        col_l, col_r = st.columns([3, 1])
        with col_l:
            fig, ax = plt.subplots(figsize=(7, 2.5))
            colors = ["#c0392b" if v > prevalence else "#2980b9" for v in prev["Prévalence (%)"]]
            ax.barh(prev["Modalité"], prev["Prévalence (%)"], color=colors, edgecolor="white")
            ax.axvline(prevalence, color="#e67e22", linestyle="--", linewidth=1.5,
                       label=f"Moy. {prevalence:.1f}%")
            for idx, row in prev.iterrows():
                ax.text(row["Prévalence (%)"] + 0.3, idx, f"{row['Prévalence (%)']:.1f}%", va="center", fontsize=9)
            ax.set_title(LABELS.get(var, var), fontweight="bold", fontsize=10)
            ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with col_r:
            sig = p < 0.05
            st.markdown(f"""
**Chi² = {chi2:.2f}**
- ddl = {dof}
- p = {'< 0,001' if p < 0.001 else f'{p:.3f}'}
- {'✅ Significatif' if sig else '❌ Non significatif'}
            """)
        st.divider()


# ══ TAB 3 — MODÈLES ML ══════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Performances comparées</div>', unsafe_allow_html=True)

    df_perf = pd.DataFrame({
        name: {k: v for k, v in vals.items() if k in ["Accuracy","AUC-ROC","Rappel","F1-score"]}
        for name, vals in results.items()
    }).T.reset_index().rename(columns={"index": "Modèle"})

    st.dataframe(
        df_perf.style.highlight_max(subset=["Accuracy","AUC-ROC","Rappel","F1-score"], color="#fde8e8")
               .format({"Accuracy":"{:.3f}","AUC-ROC":"{:.3f}","Rappel":"{:.3f}","F1-score":"{:.3f}"}),
        use_container_width=True, hide_index=True
    )

    # Barres groupées
    metrics = ["Accuracy","AUC-ROC","Rappel","F1-score"]
    x = np.arange(len(metrics)); width = 0.25
    bar_colors = ["#1a1a2e","#c0392b","#e67e22"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, color) in enumerate(zip(results.keys(), bar_colors)):
        vals_b = [results[name][m] for m in metrics]
        bars = ax.bar(x + i*width, vals_b, width, label=name, color=color, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Comparaison des modèles", fontweight="bold")
    ax.legend(); ax.spines[["top","right"]].set_visible(False)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # Rapport + matrice de confusion
    st.markdown('<div class="section-header">Rapport de classification & Matrice de confusion</div>', unsafe_allow_html=True)
    model_sel = st.selectbox("Choisir un modèle :", list(results.keys()))

    col_a, col_b = st.columns(2)
    with col_a:
        report_df = pd.DataFrame(results[model_sel]["report"]).T.round(3)
        st.dataframe(report_df, use_container_width=True)
    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, results[model_sel]["y_pred"],
            display_labels=["Pas de violence","Violence"],
            cmap="Reds", ax=ax
        )
        ax.set_title(f"Matrice — {model_sel}", fontweight="bold")
        fig.tight_layout(); st.pyplot(fig); plt.close()


# ══ TAB 4 — COURBES ROC ═════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Courbes ROC — Comparaison des modèles</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors_roc = ["#1a1a2e","#c0392b","#e67e22"]
    styles     = ["-","--","-."]
    for (name, vals), color, ls in zip(results.items(), colors_roc, styles):
        fpr, tpr, _ = roc_curve(y_test, vals["y_proba"])
        ax.plot(fpr, tpr, color=color, lw=2.5, linestyle=ls,
                label=f"{name}  (AUC = {vals['AUC-ROC']})")
    ax.plot([0,1],[0,1], "k:", lw=1, label="Modèle aléatoire (AUC = 0.5)")
    ax.fill_between(*roc_curve(y_test, results[best]["y_proba"])[:2],
                    alpha=0.07, color="#c0392b")
    ax.set_xlabel("Taux de faux positifs", fontsize=11)
    ax.set_ylabel("Taux de vrais positifs (Rappel)", fontsize=11)
    ax.set_title("Courbes ROC", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown(f"""
    **Interprétation :**
    Le modèle **{best}** obtient le meilleur AUC-ROC de **{results[best]['AUC-ROC']}**,
    indiquant une bonne capacité à discriminer les femmes victimes de violence
    de celles qui ne le sont pas. Un AUC > 0,7 est généralement considéré comme acceptable
    pour des données sociales complexes.
    """)


# ══ TAB 5 — IMPORTANCE DES VARIABLES ════════════════════════
with tab5:
    st.markdown('<div class="section-header">Importance des variables — XGBoost & Random Forest</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for col, model_name in zip([col1, col2], ["XGBoost", "Random Forest"]):
        with col:
            model = results[model_name]["model"]
            importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=True)
            median_imp = importance.median()
            bar_colors = ["#c0392b" if v >= median_imp else "#2980b9" for v in importance.values]

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(importance.index, importance.values, color=bar_colors, edgecolor="white")
            ax.set_title(f"Importance — {model_name}", fontweight="bold", fontsize=11)
            ax.set_xlabel("Importance (gain)")
            ax.axvline(median_imp, color="#e67e22", linestyle="--", linewidth=1.5,
                       label="Médiane")
            ax.legend(fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout(); st.pyplot(fig); plt.close()

            # Top 5
            top5 = importance.sort_values(ascending=False).head(5).reset_index()
            top5.columns = ["Variable", "Importance"]
            top5["Importance"] = top5["Importance"].round(4)
            st.markdown(f"**Top 5 — {model_name}**")
            st.dataframe(top5, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Interprétation</div>', unsafe_allow_html=True)
    st.markdown("""
| Facteur | Effet | Interprétation |
|---------|-------|----------------|
| **Exposition aux médias** | ⚠️ Risque accru | Les femmes exposées aux médias déclarent plus la violence (meilleure conscience ou biais de déclaration) |
| **Milieu rural** | ✅ Protecteur | OR < 1 — possible sous-déclaration ou contrôle social informel |
| **Éducation supérieure** | ✅ Protecteur | Réduit le risque de 66 % — autonomisation et connaissance des droits |
| **Éducation secondaire** | ✅ Protecteur | Réduit le risque de 23 % |
| **Alcool du conjoint** | ❌ Non significatif | Très faible prévalence déclarée (1,7 %) — probable sous-déclaration |
    """)