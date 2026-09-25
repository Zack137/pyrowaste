from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    ARTIFACT_PATH,
    DEFAULT_WEIGHTS,
    DISPLAY_NAMES,
    HALOGEN_COLUMNS,
    INPUT_COLUMNS,
    NUMERIC_INPUTS,
    OPTUNA_HYBRID_MODEL_NAME,
    OPTUNA_MLP_MODEL_NAME,
    PRODUCTION_MODEL_NAME,
    TARGET_COLUMNS,
    YIELD_COLUMNS,
)
from src.data import data_quality_report, find_data_file, load_dataset
from src.explainability import compute_shap
from src.modeling import (
    artifact_is_current,
    dense_parameter_count,
    load_bundle,
    save_bundle,
    train_all_models,
)
from src.optimization import (
    EnergyAssumptions,
    estimate_energy_cost,
    optimize_recipe,
    score_components,
)
from src.reporting import build_optimization_pdf
from src.strategies import (
    ALL_STRATEGIES,
    MODEL_STRATEGIES,
    RECOMMENDED_STRATEGY,
    STRATEGY_DESCRIPTIONS,
    evaluate_strategies,
    load_strategy_assets,
    predict_strategy,
)


st.set_page_config(
    page_title="PyroWaste",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f4f8f6 0%, #f8faf9 100%); }
      .block-container { max-width: 1450px; padding-top: 1.35rem; padding-bottom: 3rem; }
      [data-testid="stSidebar"] { background: #102a2a; }
      [data-testid="stSidebar"] * { color: #eef7f2; }
      .hero {
        position: relative; overflow: hidden; padding: 2.25rem 2.4rem; border-radius: 24px;
        background: radial-gradient(circle at 88% 16%, #d99b55 0, #d99b5500 23%),
                    linear-gradient(120deg, #0d302d 0%, #176b5b 62%, #9a6936 100%);
        color: white; margin-bottom: 1.5rem; box-shadow: 0 18px 45px #153a302c;
      }
      .hero h1 { margin: 0; font-size: 2.5rem; letter-spacing: -.045em; color: white; }
      .hero p { margin: .65rem 0 0; color: #e9f5ee; max-width: 780px; font-size: 1.04rem; }
      .hero-badge { display: inline-block; margin-bottom: .65rem; padding: .25rem .65rem;
        border: 1px solid #ffffff55; border-radius: 999px; color: #f4fbf7;
        background: #ffffff16; font-size: .76rem; letter-spacing: .08em; text-transform: uppercase; }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #e0e8e4; padding: 1rem;
        border-radius: 16px; box-shadow: 0 5px 18px #173d2d10;
      }
      div[data-testid="stMetric"]:hover { border-color: #9bc4b7; transform: translateY(-1px); }
      div[data-testid="stForm"] {
        background: white; border: 1px solid #e0e8e4; padding: 1.2rem;
        border-radius: 18px; box-shadow: 0 8px 25px #173d2d0b;
      }
      h1, h2, h3 { color: #173d38; }
      .caption-card {
        padding: .85rem 1rem; background: #edf5f1; border-left: 4px solid #2b7a67;
        border-radius: 7px; color: #234a43; margin: .5rem 0 1rem;
      }
      .workflow { display: grid; grid-template-columns: repeat(4, 1fr); gap: .65rem;
        margin: .5rem 0 1.25rem; }
      .workflow-step { background: white; border: 1px solid #dce7e2; border-radius: 13px;
        padding: .72rem .8rem; color: #687b76; font-size: .82rem; }
      .workflow-step strong { display: block; color: #38554e; font-size: .9rem; margin-top: .1rem; }
      .workflow-step.active { border-color: #2b7a67; background: #eaf5f0;
        box-shadow: inset 0 -3px 0 #2b7a67; }
      .workflow-step.done { border-color: #a8cabb; background: #f3f9f6; }
      .workflow-step.active strong, .workflow-step.done strong { color: #176b5b; }
      .section-kicker { color: #c27731; text-transform: uppercase; letter-spacing: .09em;
        font-size: .72rem; font-weight: 700; margin-bottom: .1rem; }
      .result-banner { display:flex; justify-content:space-between; align-items:center; gap:1rem;
        background: linear-gradient(100deg,#e4f3ec,#fff8ef); border:1px solid #bed9ce;
        border-radius:16px; padding:1rem 1.2rem; margin:.7rem 0 1rem; color:#173d38; }
      .result-banner .score { font-size:1.7rem; font-weight:750; color:#176b5b; white-space:nowrap; }
      .model-family-grid { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr));
        gap:.7rem; margin:.75rem 0; }
      .model-family-card { min-height:132px; padding:.9rem; border-radius:14px;
        border:1px solid #dce7e2; background:#fff; box-shadow:0 5px 18px #173d2d0c; }
      .model-family-card .family-number { color:#c27731; font-size:.7rem;
        font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
      .model-family-card strong { display:block; margin:.3rem 0; color:#176b5b;
        line-height:1.25; }
      .model-family-card span { color:#687b76; font-size:.79rem; line-height:1.35; }
      .selection-strip { padding:.8rem 1rem; border-radius:12px; text-align:center;
        border:1px solid #a8cabb; background:#eaf5f0; color:#234a43; font-size:.87rem; }
      [data-testid="stDownloadButton"] button { border-radius: 10px; font-weight: 650; }
      [data-testid="stTabs"] button { font-weight: 650; }
      @media (max-width: 1000px) { .model-family-grid { grid-template-columns:1fr 1fr; } }
      @media (max-width: 800px) {
        .workflow, .model-family-grid { grid-template-columns:1fr 1fr; }
      }
      @media (max-width: 520px) { .model-family-grid { grid-template-columns:1fr; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    return load_dataset(path)


@st.cache_resource(show_spinner=False)
def get_models(path: str, mtime: float):
    del mtime
    if artifact_is_current():
        return load_bundle()
    df = load_dataset(path)
    bundle = train_all_models(df, data_path=path)
    save_bundle(bundle)
    return bundle


@st.cache_resource(show_spinner=False)
def get_strategy_assets():
    return load_strategy_assets()


@st.cache_data(show_spinner=False)
def get_strategy_metrics(
    data_mtime: float,
    artifact_mtime: float,
):
    del data_mtime, artifact_mtime
    return evaluate_strategies(load_bundle(), load_strategy_assets())


def label(column: str) -> str:
    return DISPLAY_NAMES.get(column, column)


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <span class="hero-badge">Prototype scientifique exploratoire · E-waste pyrolysis</span>
          <h1>PyroWaste</h1>
          <p>Régression supervisée multi-sorties à huit cibles continues :
          optimiser séparément Ridge polynomial, Extra Trees, un MLP compact et
          un hybride à deux branches MLP + XGBoost avec Optuna, puis
          présélectionner une famille pour chaque sortie.
          Aucune décision industrielle autonome.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_steps(active: int) -> None:
    labels = [
        ("01", "Configurer"),
        ("02", "Prédire"),
        ("03", "Optimiser"),
        ("04", "Rapport"),
    ]
    cards = []
    for index, (number, title) in enumerate(labels, start=1):
        state = "active" if index == active else "done" if index < active else ""
        cards.append(
            f'<div class="workflow-step {state}"><span>ÉTAPE {number}</span>'
            f"<strong>{title}</strong></div>"
        )
    st.markdown('<div class="workflow">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_overview(df: pd.DataFrame, bundle) -> None:
    st.subheader("Vue d’ensemble expérimentale")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expériences", f"{len(df):,}".replace(",", " "))
    gas_columns = [
        column
        for column in df.columns
        if column.endswith("_ppb") or column.endswith("_ppm")
    ]
    c2.metric("Mesures gazeuses enrichies", len(gas_columns))
    c3.metric("Variables d’entrée", len(INPUT_COLUMNS))
    c4.metric("Stratégies disponibles", len(ALL_STRATEGIES))

    st.markdown(
        '<div class="caption-card">Les rendements sont contraints à totaliser 100 %. '
        "L’indice de toxicité et le coût du liquide sont calculés, pas appris.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Architecture scientifique active", expanded=True):
        st.code(
            """
BASE PYROLYSE
      │
      ▼
PRÉTRAITEMENT → FEATURE ENGINEERING SANS CIBLE
      │
      ├── Ridge polynomial ───────────── Optuna ──┐
      ├── Extra Trees ────────────────── Optuna ──┤
      ├── MLP compact ────────────────── Optuna ──┼─→ meilleure famille par sortie
      └── Hybride 2 branches MLP + XGBoost ─ Optuna ┘
                                                  │
                                                  ▼
                                R² · RMSE · MAE · IC95 par sortie
                                                  │
                                                  ▼
                      cohérence numérique → prédictions finales
            """.strip(),
            language="text",
        )
        st.caption(
            "Random Forest, l’ancien XGBoost autonome et le MLP sans Optuna "
            "restent supprimés. XGBoost n’existe que comme branche interne du "
            "nouvel hybride optimisé par Optuna."
        )
        st.markdown(
            """
            <div class="model-family-grid">
              <div class="model-family-card">
                <div class="family-number">Famille 01</div>
                <strong>Ridge polynomial Optuna</strong>
                <span>Référence linéaire régularisée sur les variables enrichies.</span>
              </div>
              <div class="model-family-card">
                <div class="family-number">Famille 02</div>
                <strong>Extra Trees Optuna</strong>
                <span>Ensemble d’arbres aléatoires optimisé séparément.</span>
              </div>
              <div class="model-family-card">
                <div class="family-number">Famille 03</div>
                <strong>MLP Optuna compact</strong>
                <span>Réseau réduit pour limiter la surparamétrisation.</span>
              </div>
              <div class="model-family-card">
                <div class="family-number">Famille 04</div>
                <strong>Hybride MLP + XGBoost Optuna</strong>
                <span>Deux branches complémentaires combinées pour chaque sortie.</span>
              </div>
            </div>
            <div class="selection-strip">
              Entraînement séparé → évaluation séparée → sélection automatique
              de la meilleure famille pour chaque cible
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.15, 0.85])
    with left:
        selected = st.selectbox(
            "Variable à explorer",
            NUMERIC_INPUTS + TARGET_COLUMNS,
            format_func=label,
        )
        figure = px.histogram(
            df,
            x=selected,
            nbins=35,
            title=f"Distribution — {label(selected)}",
            template="plotly_white",
        )
        figure.update_layout(showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(figure, width="stretch")
    with right:
        averages = df[TARGET_COLUMNS].mean().reset_index()
        averages.columns = ["Sortie", "Moyenne"]
        averages["Libellé"] = averages["Sortie"].map(label)
        figure = px.bar(
            averages,
            x="Moyenne",
            y="Libellé",
            orientation="h",
            title="Moyenne des sorties expérimentales",
            color_discrete_sequence=["#2b7a67"],
            template="plotly_white",
        )
        figure.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(figure, width="stretch")

    with st.expander("Qualité et structure de la base"):
        st.dataframe(data_quality_report(df), width="stretch", hide_index=True)

    with st.expander("Feature engineering commun aux huit sorties"):
        st.caption(
            "L’utilisateur saisit les cinq entrées numériques. Chaque famille "
            "construit ensuite les mêmes variables sans utiliser les sorties."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    ["Interaction température–temps", "Température × temps de réaction"],
                    ["Interaction vitesse de chauffage–temps", "Vitesse de chauffe × temps de réaction"],
                    ["Température au carré", "Température²"],
                    ["Interaction température–vitesse de chauffage", "Température × vitesse de chauffe"],
                    ["Rapport température–particules", "Température / taille des particules"],
                    ["Temps logarithmique", "ln(1 + temps de réaction)"],
                    ["Rapport quantité–taille", "Quantité / taille des particules"],
                    ["Interaction quantité–température", "Quantité × température"],
                    ["Interaction quantité–temps", "Quantité × temps"],
                    ["Interaction quantité–vitesse de chauffage", "Quantité × vitesse de chauffe"],
                ],
                columns=["Variable enrichie", "Calcul"],
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Le modèle reçoit 15 caractéristiques : 5 entrées brutes et 10 "
            "variables dérivées. Aucune sortie expérimentale n’est utilisée comme entrée."
        )


def _r2_interpretation(value: float) -> str:
    if value < 0:
        return "Inférieur au prédicteur moyen"
    return "Positif; à interpréter avec le gain vs prédicteur naïf"


def render_models(
    summary: pd.DataFrame,
    target_metrics: pd.DataFrame,
    df: pd.DataFrame,
    bundle,
) -> None:
    st.subheader("Pilotage interne de l’artefact exploratoire")
    st.caption(
        "Ce découpage unique 80/10/10 sert à construire et contrôler l’artefact "
        "interactif. Il fournit une estimation interne seulement ; la preuve "
        "scientifique principale est la validation croisée répétée et imbriquée."
    )
    validation = summary[summary["Jeu"] == "Validation"].sort_values(
        "RMSE normalisé"
    )
    test = summary[summary["Jeu"] == "Test"].sort_values("RMSE normalisé")
    tabs = st.tabs(["Validation — présélection", "Test — estimation interne"])
    for tab, frame in zip(tabs, [validation, test]):
        with tab:
            st.dataframe(
                frame.style.format(
                    {
                        "MAE normalisée": "{:.4f}",
                        "RMSE normalisé": "{:.4f}",
                        "R² macro": "{:.4f}",
                        "Temps entraînement (s)": "{:.1f}",
                    }
                ).highlight_min(subset=["RMSE normalisé"], color="#d9f0e5"),
                width="stretch",
                hide_index=True,
            )

    best_validation = validation.iloc[0]
    recommended_rows = test[test["Stratégie"] == RECOMMENDED_STRATEGY]
    recommended_test = (
        recommended_rows.iloc[0] if not recommended_rows.empty else test.iloc[0]
    )
    a, b, c, d = st.columns(4)
    a.metric("Meilleure validation", str(best_validation["Stratégie"]))
    b.metric("RMSE validation", f"{best_validation['RMSE normalisé']:.3f}")
    c.metric(
        "RMSE test présélectionné",
        f"{recommended_test['RMSE normalisé']:.3f}",
    )
    d.metric(
        "R² test présélectionné",
        f"{recommended_test['R² macro']:.3f}",
    )

    st.warning(
        "Les émissions sont bien expliquées, mais les R² des rendements restent "
        "faibles. Toute proposition doit être validée expérimentalement avant usage."
    )
    st.markdown("#### Modèle retenu pour chaque sortie")
    selection = pd.DataFrame(
        [
            {
                "Sortie": label(target),
                "Famille retenue sur validation": family,
            }
            for target, family in bundle.best_model_by_target.items()
        ]
    )
    st.dataframe(selection, width="stretch", hide_index=True)
    st.caption(
        "Ce mapping est fixé avant l'évaluation du test. Le post-traitement "
        "physique est appliqué après l'assemblage des huit prédictions."
    )

    with st.expander(
        "Complexité des branches MLP Optuna — règle de 5 à 10 observations "
        "par paramètre"
    ):
        complexity = []
        train_observations = len(bundle.splits.X_train)
        neural_components = [
            (OPTUNA_MLP_MODEL_NAME, "MLP autonome"),
            (OPTUNA_HYBRID_MODEL_NAME, "Branche MLP de l’hybride"),
        ]
        for family, role in neural_components:
            family_params = bundle.optuna_best_params.get(family, {})
            for target in TARGET_COLUMNS:
                params = family_params.get(target)
                if not params:
                    continue
                layer_count = int(params.get("n_layers", 1))
                layers = tuple(
                    int(params.get(f"units_layer_{index}", 8))
                    for index in range(1, layer_count + 1)
                )
                parameters = dense_parameter_count(15, layers, 1)
                complexity.append(
                    {
                        "Composant": role,
                        "Sortie": label(target),
                        "Couches cachées": "–".join(map(str, layers)),
                        "Paramètres": parameters,
                        "Observations/paramètre": (
                            train_observations / parameters
                        ),
                        "Minimum indicatif (5×)": 5 * parameters,
                        "Minimum prudent (10×)": 10 * parameters,
                        f"Compatible avec {train_observations}": (
                            "Oui"
                            if train_observations >= 5 * parameters
                            else "Non"
                        ),
                    }
                )
        st.dataframe(
            pd.DataFrame(complexity).style.format(
                {"Observations/paramètre": "{:.3f}"}
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Le MLP autonome et la branche neuronale de l’hybride restent "
            "compacts. La complexité de la branche XGBoost est contrôlée "
            "séparément par ses hyperparamètres d’arbres."
        )

    st.subheader("Comparaison par sortie sur le test")
    target = st.selectbox("Sortie", TARGET_COLUMNS, format_func=label, key="metric_target")
    details = target_metrics[
        (target_metrics["Sortie"] == target)
        & (target_metrics["Jeu"] == "Test")
    ][["Stratégie", "MAE", "RMSE", "R²"]].copy()
    output_std = float(df[target].std(ddof=0))
    output_range = float(df[target].max() - df[target].min())
    baseline_rows = details[
        details["Stratégie"] == "Baseline moyenne (référence)"
    ]
    baseline_rmse = (
        float(baseline_rows["RMSE"].iloc[0])
        if not baseline_rows.empty
        else float("nan")
    )
    details["RMSE / écart-type"] = details["RMSE"] / max(output_std, 1e-12)
    details["RMSE / étendue"] = details["RMSE"] / max(output_range, 1e-12)
    details["Gain RMSE vs moyenne (%)"] = (
        (baseline_rmse - details["RMSE"]) / baseline_rmse * 100.0
    )
    details["Interprétation R²"] = details["R²"].map(_r2_interpretation)
    details = details.sort_values("RMSE")
    st.dataframe(
        details.style.format(
            {
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "R²": "{:.4f}",
                "RMSE / écart-type": "{:.3f}",
                "RMSE / étendue": "{:.3f}",
                "Gain RMSE vs moyenne (%)": "{:+.1f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Échelle de {label(target)} : moyenne {df[target].mean():.3f}, "
        f"écart-type {output_std:.3f}, étendue {output_range:.3f}. "
        "Un R² proche de zéro apporte peu par rapport à la moyenne; un R² négatif "
        "est inférieur à ce prédicteur naïf."
    )
    chart = px.bar(
        details,
        x="Stratégie",
        y="RMSE",
        color="Stratégie",
        title=f"RMSE test — {label(target)}",
        template="plotly_white",
    )
    chart.update_layout(showlegend=False)
    st.plotly_chart(chart, width="stretch")


def recipe_form(df: pd.DataFrame, bundle) -> tuple[bool, pd.DataFrame]:
    defaults = df[NUMERIC_INPUTS].median()
    with st.form("prediction_form"):
        st.markdown("#### Conditions opératoires")
        columns = st.columns(3)
        values: dict[str, float] = {}
        for index, column in enumerate(NUMERIC_INPUTS):
            low, high = bundle.input_bounds[column]
            span = max(high - low, 0.01)
            values[column] = columns[index % 3].number_input(
                label(column),
                min_value=float(low),
                max_value=float(high),
                value=float(defaults[column]),
                step=float(max(span / 100.0, 0.01)),
            )
        submitted = st.form_submit_button("Lancer la prédiction", type="primary")
    return submitted, pd.DataFrame([values], columns=INPUT_COLUMNS)


def render_prediction(df: pd.DataFrame, bundle, strategy_assets) -> None:
    st.markdown('<div class="section-kicker">Laboratoire de scénarios</div>', unsafe_allow_html=True)
    st.subheader("Prédiction et optimisation d’un scénario")
    selected_strategy = st.selectbox(
        "Stratégie de modèle",
        ALL_STRATEGIES,
        index=0,
        help=(
            "La première stratégie est présélectionnée sur le jeu de validation "
            "interne ; ce choix reste exploratoire."
        ),
    )
    st.caption(STRATEGY_DESCRIPTIONS[selected_strategy])
    submitted, inputs = recipe_form(df, bundle)
    if submitted:
        st.session_state["prediction_inputs"] = inputs.copy()
        st.session_state["prediction_strategy"] = selected_strategy
        st.session_state.pop("post_prediction_result", None)
        st.session_state.pop("post_prediction_trials", None)
    elif "prediction_inputs" not in st.session_state:
        workflow_steps(1)
        st.info("Saisissez les conditions puis lancez la prédiction.")
        return

    inputs = st.session_state["prediction_inputs"].copy()
    active_strategy = st.session_state.get("prediction_strategy", RECOMMENDED_STRATEGY)
    optimization_result = st.session_state.get("post_prediction_result")
    workflow_steps(4 if optimization_result is not None else 2)
    if active_strategy != selected_strategy:
        st.info(
            f"Résultat affiché : {active_strategy}. Cliquez sur « Lancer la prédiction » "
            "pour appliquer la nouvelle stratégie sélectionnée."
        )

    prediction = predict_strategy(bundle, strategy_assets, inputs, active_strategy)
    assumptions = EnergyAssumptions()
    score = score_components(inputs, prediction, df, assumptions=assumptions)
    energy_cost = estimate_energy_cost(inputs, assumptions).iloc[0]

    st.markdown("#### Résultats prédits")
    st.markdown(f"**Stratégie active :** {active_strategy}")
    columns = st.columns(3)
    for column, container in zip(YIELD_COLUMNS, columns):
        container.metric(label(column), f"{prediction[column].iloc[0]:.2f} %")

    visual_left, visual_middle, visual_right = st.columns([0.9, 1.25, 0.85])
    with visual_left:
        yield_frame = pd.DataFrame(
            {
                "Phase": ["Solide", "Liquide", "Gaz"],
                "Rendement": [float(prediction[column].iloc[0]) for column in YIELD_COLUMNS],
            }
        )
        yield_chart = px.pie(
            yield_frame,
            values="Rendement",
            names="Phase",
            hole=0.58,
            title="Bilan des rendements",
            color="Phase",
            color_discrete_map={"Solide": "#557A72", "Liquide": "#D18A3A", "Gaz": "#62A68F"},
        )
        yield_chart.update_traces(textinfo="label+percent", textposition="outside")
        yield_chart.update_layout(height=315, margin=dict(l=10, r=10, t=55, b=10), showlegend=False)
        st.plotly_chart(yield_chart, width="stretch")
    with visual_middle:
        emissions = prediction[HALOGEN_COLUMNS].T.reset_index()
        emissions.columns = ["Émission", "Valeur prédite (%)"]
        emissions["Émission"] = emissions["Émission"].map(label)
        chart = px.bar(
            emissions,
            x="Émission",
            y="Valeur prédite (%)",
            color_discrete_sequence=["#c56d32"],
            template="plotly_white",
            title="Émissions halogénées",
        )
        chart.update_layout(height=315, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(chart, width="stretch")
    with visual_right:
        score_value = float(score["Score_final"].iloc[0])
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score_value,
                number={"suffix": "/100", "font": {"size": 27, "color": "#173D38"}},
                title={"text": "Score multicritère", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#176B5B"},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 50], "color": "#F5DDD8"},
                        {"range": [50, 75], "color": "#F5E9D4"},
                        {"range": [75, 100], "color": "#DDEFE7"},
                    ],
                },
            )
        )
        gauge.update_layout(height=205, margin=dict(l=18, r=18, t=45, b=5))
        st.plotly_chart(gauge, width="stretch")
        st.metric("Indice de toxicité", f"{prediction['Toxicity_Index_pct'].iloc[0]:.3f} %")
        st.metric("Coût du liquide", f"{prediction['Cout_liquide_USD'].iloc[0]:,.2f} $")
        st.caption(f"Coût énergétique estimé : {energy_cost:,.2f} $")

    st.divider()
    st.markdown("#### Optimisation après la prédiction")
    st.caption(
        "Optuna explore de nouvelles conditions dans le domaine expérimental. "
        "Chaque candidat est évalué par la stratégie active."
    )
    st.warning(
        "Scénario exploratoire uniquement : les performances des rendements "
        "restent limitées dans la validation imbriquée. L'optimiseur peut "
        "exploiter une erreur du modèle; toute condition proposée doit être "
        "vérifiée par une expérience réelle."
    )
    with st.form("post_prediction_optimization"):
        trials = st.slider("Essais Optuna", 50, 250, 100, 25)
        optimize_submitted = st.form_submit_button(
            "Optimiser après cette prédiction", type="primary"
        )

    with st.expander("Priorités et hypothèses utilisées"):
        weights_table = pd.DataFrame(
            {
                "Objectif": [label(key) for key in DEFAULT_WEIGHTS],
                "Poids": list(DEFAULT_WEIGHTS.values()),
            }
        ).sort_values("Poids", ascending=False)
        st.dataframe(weights_table, width="stretch", hide_index=True)
        st.caption(
            "Priorité principale : minimisation du liquide chimique (quantité + coût), "
            "puis minimisation de la toxicité. Coût énergétique calculé avec un "
            "réacteur de 10 kW et une électricité à 0,15 $/kWh."
        )

    if optimize_submitted:
        with st.spinner(f"Optimisation de {trials} scénarios — {active_strategy}…"):
            def strategy_predictor(candidate):
                return predict_strategy(
                    bundle,
                    strategy_assets,
                    candidate,
                    active_strategy,
                )

            st.session_state["post_prediction_result"] = optimize_recipe(
                bundle,
                n_trials=trials,
                assumptions=assumptions,
                predictor=strategy_predictor,
            )
            st.session_state["post_prediction_trials"] = int(trials)
        st.rerun()

    optimization_result = st.session_state.get("post_prediction_result")
    if optimization_result is None:
        return

    optimum = optimization_result.recommendation.iloc[0]
    before_record = {
        **{column: float(inputs[column].iloc[0]) for column in NUMERIC_INPUTS},
        **{column: float(prediction[column].iloc[0]) for column in TARGET_COLUMNS},
        "Toxicity_Index_pct": float(prediction["Toxicity_Index_pct"].iloc[0]),
        "Cout_liquide_USD": float(prediction["Cout_liquide_USD"].iloc[0]),
        "Cout_energie_USD": float(energy_cost),
        "Score_final": float(score["Score_final"].iloc[0]),
    }
    after_record = {key: value for key, value in optimum.to_dict().items()}
    used_trials = int(st.session_state.get("post_prediction_trials", trials))

    score_delta = float(after_record["Score_final"]) - float(before_record["Score_final"])
    st.markdown(
        f"""
        <div class="result-banner">
          <div><strong>Scénario exploratoire Optuna</strong><br>
          <span>{active_strategy} · {used_trials} essais · modèle à 5 variables</span></div>
          <div class="score">{float(after_record['Score_final']):.1f}/100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Score prédit du scénario",
        f"{float(after_record['Score_final']):.1f}/100",
        f"{score_delta:+.1f} points",
    )
    k2.metric(
        "Liquide chimique",
        f"{float(after_record['Concentration_liquide_chimique_kg']):.2f} kg",
        f"{float(after_record['Concentration_liquide_chimique_kg'])-float(before_record['Concentration_liquide_chimique_kg']):+.2f} kg",
        delta_color="inverse",
    )
    k3.metric(
        "Toxicity Index",
        f"{float(after_record['Toxicity_Index_pct']):.3f} %",
        f"{float(after_record['Toxicity_Index_pct'])-float(before_record['Toxicity_Index_pct']):+.3f} %",
        delta_color="inverse",
    )
    k4.metric(
        "Coût liquide",
        f"{float(after_record['Cout_liquide_USD']):,.2f} $",
        f"{float(after_record['Cout_liquide_USD'])-float(before_record['Cout_liquide_USD']):+,.2f} $",
        delta_color="inverse",
    )

    chart_tabs = st.tabs(
        ["Rendements", "Émissions", "Conditions normalisées", "Convergence Optuna"]
    )
    with chart_tabs[0]:
        yield_comparison = pd.DataFrame(
            [
                {
                    "Phase": phase,
                    "Moment": moment,
                    "Rendement (%)": record[column],
                }
                for phase, column in zip(["Solide", "Liquide", "Gaz"], YIELD_COLUMNS)
                for moment, record in [("Avant", before_record), ("Après", after_record)]
            ]
        )
        figure = px.bar(
            yield_comparison,
            x="Phase",
            y="Rendement (%)",
            color="Moment",
            barmode="group",
            color_discrete_map={"Avant": "#93A9A3", "Après": "#176B5B"},
            template="plotly_white",
            title="Rendements avant et après optimisation",
        )
        figure.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(figure, width="stretch")
    with chart_tabs[1]:
        gas_comparison = pd.DataFrame(
            [
                {
                    "Gaz": label(column),
                    "Moment": moment,
                    "Concentration (%)": record[column],
                }
                for column in HALOGEN_COLUMNS
                for moment, record in [("Avant", before_record), ("Après", after_record)]
            ]
        )
        figure = px.bar(
            gas_comparison,
            x="Gaz",
            y="Concentration (%)",
            color="Moment",
            barmode="group",
            color_discrete_map={"Avant": "#D3A476", "Après": "#B65338"},
            template="plotly_white",
            title="Émissions halogénées avant et après",
        )
        figure.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(figure, width="stretch")
    with chart_tabs[2]:
        normalized_rows = []
        for column in NUMERIC_INPUTS:
            low, high = bundle.input_bounds[column]
            span = max(high - low, 1e-12)
            for moment, record in [("Avant", before_record), ("Après", after_record)]:
                normalized_rows.append(
                    {
                        "Condition": label(column),
                        "Moment": moment,
                        "Position dans la plage (%)": 100.0
                        * (float(record[column]) - low)
                        / span,
                    }
                )
        normalized = pd.DataFrame(normalized_rows)
        figure = px.bar(
            normalized,
            y="Condition",
            x="Position dans la plage (%)",
            color="Moment",
            barmode="group",
            orientation="h",
            color_discrete_map={"Avant": "#93A9A3", "Après": "#D18A3A"},
            template="plotly_white",
            title="Déplacement des conditions dans le domaine expérimental",
        )
        figure.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(figure, width="stretch")
    with chart_tabs[3]:
        figure = px.line(
            optimization_result.history,
            x="Essai",
            y="Meilleur score cumulé",
            markers=True,
            template="plotly_white",
            color_discrete_sequence=["#176B5B"],
            title="Convergence de la recherche Optuna",
        )
        figure.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(figure, width="stretch")

    comparison_specs = [
        ("Liquide chimique (kg)", "Concentration_liquide_chimique_kg", "min"),
        ("Température (°C)", "Temperature_C", "neutral"),
        ("Temps de réaction (min)", "Temps_reaction_min", "min"),
        ("Heating rate (°C/min)", "Heating_rate_C_min", "neutral"),
        ("Taille particules (mm)", "Taille_particules_mm", "neutral"),
        ("Solid Yield (%)", "Solid_Yield_pct", "max"),
        ("Liquid Yield (%)", "Liquid_Yield_pct", "max"),
        ("Gas Yield (%)", "Gas_Yield_pct", "max"),
        ("HBr (%)", "HBr_pct", "min"),
        ("Br2 (%)", "Br2_pct", "min"),
        ("HCl (%)", "HCl_pct", "min"),
        ("Cl2 (%)", "Cl2_pct", "min"),
        ("HF (%)", "HF_pct", "min"),
        ("Toxicity Index (%)", "Toxicity_Index_pct", "min"),
        ("Coût liquide ($)", "Cout_liquide_USD", "min"),
        ("Coût énergétique ($)", "Cout_energie_USD", "min"),
        ("Score final", "Score_final", "max"),
    ]
    comparison_rows = []
    for indicator, key, direction in comparison_specs:
        old, new = float(before_record[key]), float(after_record[key])
        change = new - old
        variation = 0.0 if abs(old) < 1e-12 else change / abs(old) * 100.0
        if direction == "neutral":
            interpretation = "Ajustement"
        else:
            improved = new < old if direction == "min" else new > old
            interpretation = "Amélioration" if improved else "Compromis"
        comparison_rows.append(
            {
                "Indicateur": indicator,
                "Avant optimisation": old,
                "Après optimisation": new,
                "Écart": change,
                "Variation (%)": variation,
                "Lecture": interpretation,
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    st.markdown("#### Tableau détaillé avant / après")
    st.dataframe(
        comparison.style.format(
            {
                "Avant optimisation": "{:.3f}",
                "Après optimisation": "{:.3f}",
                "Écart": "{:+.3f}",
                "Variation (%)": "{:+.1f} %",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    pdf_bytes = build_optimization_pdf(
        strategy=active_strategy,
        before=before_record,
        after=after_record,
        trials=used_trials,
        selected_models_by_target=(
            bundle.best_model_by_target
            if active_strategy == PRODUCTION_MODEL_NAME
            else None
        ),
    )
    safe_configuration = "5_variables"
    download_csv, download_pdf = st.columns(2)
    download_csv.download_button(
        "Télécharger le tableau CSV",
        comparison.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"comparaison_{safe_configuration}.csv",
        mime="text/csv",
        key="download_post_optimization_csv",
        width="stretch",
    )
    download_pdf.download_button(
        "Télécharger le rapport PDF",
        pdf_bytes,
        file_name=f"rapport_pyrowaste_{safe_configuration}.pdf",
        mime="application/pdf",
        key="download_post_optimization_pdf",
        width="stretch",
        type="primary",
    )


def render_shap(bundle) -> None:
    st.subheader("Explicabilité SHAP")
    st.caption(
        "Les contributions sont calculées sur la famille effectivement retenue "
        "pour la sortie choisie et exprimées dans son unité physique."
    )
    target = st.selectbox("Sortie à expliquer", TARGET_COLUMNS, format_func=label, key="shap_target")
    if st.button("Calculer les explications SHAP", type="primary"):
        with st.spinner("Calcul des contributions SHAP…"):
            st.session_state["shap_result"] = compute_shap(bundle, target)

    result = st.session_state.get("shap_result")
    if result is None or result.target != target:
        st.info("Choisissez une sortie puis lancez le calcul SHAP.")
        return
    st.success(f"Modèle expliqué : {result.model_name}")

    importance = result.importance.sort_values("Importance SHAP moyenne")
    importance["Libellé"] = importance["Variable"].map(label)
    chart = px.bar(
        importance,
        x="Importance SHAP moyenne",
        y="Libellé",
        orientation="h",
        color_discrete_sequence=["#2b7a67"],
        template="plotly_white",
        title=f"Variables déterminantes — {label(target)}",
    )
    st.plotly_chart(chart, width="stretch")

    st.markdown("#### Effets des variables opératoires")
    tabs = st.tabs(["Liquide chimique", "Température", "Temps de réaction"])
    features = [
        "Concentration_liquide_chimique_kg",
        "Temperature_C",
        "Temps_reaction_min",
    ]
    for tab, feature in zip(tabs, features):
        with tab:
            effect = result.effects[[feature, f"SHAP__{feature}"]].copy()
            figure = px.scatter(
                effect,
                x=feature,
                y=f"SHAP__{feature}",
                labels={
                    feature: label(feature),
                    f"SHAP__{feature}": f"Effet SHAP sur {label(target)}",
                },
                template="plotly_white",
            )
            st.plotly_chart(figure, width="stretch")


def _render_professor_cross_validation_legacy() -> None:
    validation_dir = (
        Path(__file__).resolve().parent / "reports" / "professor_validation"
    )
    cv_path = validation_dir / "cv_summary.csv"
    metadata_path = validation_dir / "run_metadata.json"
    augmentation_path = validation_dir / "augmentation_summary.csv"

    st.divider()
    st.markdown("#### Validation croisée indépendante demandée par le professeur")
    if not cv_path.exists():
        st.info("Les résultats de validation croisée ne sont pas disponibles.")
        return

    cv_all = pd.read_csv(cv_path, encoding="utf-8-sig")
    global_rows = cv_all[cv_all["Sortie"] == "GLOBAL_macro"].copy()
    global_rows = global_rows.sort_values("NRMSE_std_moyenne")
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Plis exécutés", int(metadata.get("splits_per_model", 0)))
    c2.metric("Meilleur modèle CV", str(global_rows.iloc[0]["Modele"]))
    c3.metric("R² macro CV", f"{global_rows.iloc[0]['R2_moyenne']:.3f}")
    st.caption(
        f"Protocole réellement exécuté : {metadata.get('folds', '—')} plis × "
        f"{metadata.get('repeats', '—')} répétition, mêmes partitions pour tous "
        "les modèles. Les IC95 sur cinq plis sont descriptifs."
    )
    cv_display = global_rows[
        [
            "Modele",
            "NRMSE_std_moyenne",
            "NRMSE_std_ecart_type",
            "R2_moyenne",
            "R2_ecart_type",
            "Amelioration_RMSE_vs_Dummy_pct_moyenne",
            "N_splits",
        ]
    ].rename(
        columns={
            "Modele": "Modèle",
            "NRMSE_std_moyenne": "NRMSE/écart-type",
            "NRMSE_std_ecart_type": "Écart-type NRMSE",
            "R2_moyenne": "R² moyen",
            "R2_ecart_type": "Écart-type R²",
            "Amelioration_RMSE_vs_Dummy_pct_moyenne": "Gain vs moyenne (%)",
            "N_splits": "Plis",
        }
    )
    st.dataframe(
        cv_display.style.format(
            {
                "NRMSE/écart-type": "{:.4f}",
                "Écart-type NRMSE": "{:.4f}",
                "R² moyen": "{:.4f}",
                "Écart-type R²": "{:.4f}",
                "Gain vs moyenne (%)": "{:+.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    error = (
        global_rows["NRMSE_std_IC95_haut"]
        - global_rows["NRMSE_std_moyenne"]
    ).clip(lower=0.0)
    figure = px.bar(
        global_rows,
        x="Modele",
        y="NRMSE_std_moyenne",
        error_y=error,
        labels={
            "Modele": "Modèle",
            "NRMSE_std_moyenne": "NRMSE / écart-type",
        },
        title="Validation croisée — NRMSE et IC95",
        template="plotly_white",
    )
    figure.update_layout(showlegend=False)
    st.plotly_chart(figure, width="stretch")

    selected_target = st.selectbox(
        "Sortie détaillée en validation croisée",
        TARGET_COLUMNS,
        format_func=label,
        key="professor_cv_target",
    )
    target_rows = cv_all[cv_all["Sortie"] == selected_target].sort_values(
        "RMSE_moyenne"
    )
    st.dataframe(
        target_rows[
            [
                "Modele",
                "MAE_moyenne",
                "RMSE_moyenne",
                "R2_moyenne",
                "R2_IC95_bas",
                "R2_IC95_haut",
                "Amelioration_RMSE_vs_Dummy_pct_moyenne",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Augmentation contrôlée, appliquée au train seulement")
    if augmentation_path.exists():
        augmentation = pd.read_csv(
            augmentation_path, encoding="utf-8-sig"
        )
        st.dataframe(
            augmentation.style.format(
                {
                    "MAE_normalisee": "{:.4f}",
                    "RMSE_normalise": "{:.4f}",
                    "R2_macro": "{:.4f}",
                    "Temps_entrainement_s": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        validation_rows = augmentation[augmentation["Jeu"] == "Validation"]
        st.plotly_chart(
            px.bar(
                validation_rows,
                x="Strategie",
                y="RMSE_normalise",
                color="Architecture",
                title="Avant/après augmentation — validation",
                template="plotly_white",
            ),
            width="stretch",
        )
        st.warning(
            "Bootstrap, jitter et Mixup sont des régularisations; ils ne créent "
            "aucune nouvelle preuve expérimentale."
        )

    diagnostics_dir = validation_dir / "diagnostics"
    images = [
        ("observed_vs_predicted_ridge.png", "Observé–prédit — Ridge"),
        ("residual_distributions_ridge.png", "Distribution des résidus — Ridge"),
        (
            "residuals_vs_inputs_ridge.png",
            "Résidus des rendements selon les entrées — Ridge",
        ),
        (
            "learning_curves_ridge_mlp_compact.png",
            "Courbes d’apprentissage Ridge et MLP compact",
        ),
    ]
    available = [
        (diagnostics_dir / name, caption)
        for name, caption in images
        if (diagnostics_dir / name).exists()
    ]
    if available:
        tabs = st.tabs([caption for _, caption in available])
        for tab, (path, caption) in zip(tabs, available):
            with tab:
                st.image(
                    str(path),
                    caption=caption,
                    width="stretch",
                )
    st.info(
        "VAR n’est pas appliquée : les lignes sont des expériences indépendantes "
        "sans ordre temporel ni retards. Kernel Ridge RBF et SVR-RBF constituent "
        "les références à noyau adaptées à ces données tabulaires."
    )


def _render_scientific_validation_legacy(bundle) -> None:
    st.subheader("Validation scientifique — protocole sans fuite du test")
    protocol = bundle.optuna_protocol
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entraînement", len(bundle.splits.X_train))
    c2.metric("Validation", len(bundle.splits.X_validation))
    c3.metric("Test final", len(bundle.splits.X_test))
    c4.metric(
        "Essais Optuna / cible",
        int(protocol["trials_per_family_and_target"]),
    )
    st.info(
        f"Optuna : {protocol['inner_validation']}. "
        f"Sélection : {protocol['target_selection']}. "
        f"Test : {protocol['test_role']}."
    )

    st.markdown("#### Traçabilité de la sélection par sortie")
    rows = []
    validation_metrics = bundle.target_metrics[
        bundle.target_metrics["Jeu"] == "Validation"
    ]
    test_metrics = bundle.target_metrics[
        bundle.target_metrics["Jeu"] == "Test"
    ]
    for target, family in bundle.best_model_by_target.items():
        validation_row = validation_metrics[
            (validation_metrics["Sortie"] == target)
            & (validation_metrics["Modèle"] == family)
        ].iloc[0]
        test_row = test_metrics[
            (test_metrics["Sortie"] == target)
            & (test_metrics["Modèle"] == PRODUCTION_MODEL_NAME)
        ].iloc[0]
        rows.append(
            {
                "Sortie": label(target),
                "Famille choisie": family,
                "RMSE validation": validation_row["RMSE"],
                "R² validation": validation_row["R²"],
                "RMSE test final": test_row["RMSE"],
                "R² test final": test_row["R²"],
            }
        )
    selection = pd.DataFrame(rows)
    st.dataframe(
        selection.style.format(
            {
                "RMSE validation": "{:.4f}",
                "R² validation": "{:.4f}",
                "RMSE test final": "{:.4f}",
                "R² test final": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Historique des optimisations d'hyperparamètres")
    trials = bundle.optuna_trials.copy()
    completed = trials[trials["Etat"] == "COMPLETE"]
    summary = (
        completed.groupby(["Famille", "Sortie"], as_index=False)
        .agg(
            Essais=("Essai", "count"),
            Meilleure_RMSE_CV=("RMSE_CV_normalisee", "min"),
        )
        .sort_values(["Sortie", "Meilleure_RMSE_CV"])
    )
    st.dataframe(
        summary.style.format({"Meilleure_RMSE_CV": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )
    with st.expander("Meilleurs hyperparamètres par famille et sortie"):
        parameter_rows = []
        for family, targets in bundle.optuna_best_params.items():
            for target, params in targets.items():
                parameter_rows.append(
                    {
                        "Famille": family,
                        "Sortie": label(target),
                        "Hyperparamètres": str(params),
                    }
                )
        st.dataframe(
            pd.DataFrame(parameter_rows),
            width="stretch",
            hide_index=True,
        )

    st.warning(
        "Le jeu de test est isolé de l'optimisation et de la sélection, mais il "
        "provient de la même base de 1 000 observations. Une campagne externe "
        "reste nécessaire avant toute conclusion industrielle."
    )
    st.info(
        "La comparaison scientifique complète — 5 plis × 5 répétitions, "
        "intervalles à 95 %, références classiques, modèles lourds et diagnostics "
        "hors pli — se trouve dans la page « Validation scientifique »."
    )


SCIENTIFIC_REPORT_DIR = (
    Path(__file__).resolve().parent / "reports" / "professor_validation"
)
NESTED_VALIDATION_DIR = (
    Path(__file__).resolve().parent
    / "reports"
    / "nested_repeated_validation_budget_renforce"
)
OPTUNA_EVIDENCE_DIR = (
    Path(__file__).resolve().parent / "reports" / "optuna_scientific_evidence"
)


def _science_csv(filename: str) -> pd.DataFrame | None:
    """Charge un CSV scientifique sans bloquer les autres sections."""
    path = SCIENTIFIC_REPORT_DIR / filename
    if not path.exists():
        st.info(f"Résultat non disponible : {filename}")
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        st.warning(f"Impossible de lire {filename} : {exc}")
        return None


def _science_json(filename: str) -> dict:
    path = SCIENTIFIC_REPORT_DIR / filename
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Impossible de lire {filename} : {exc}")
        return {}


def _result_csv(directory: Path, filename: str) -> pd.DataFrame | None:
    """Charge un résultat tabulaire depuis un dossier scientifique donné."""
    path = directory / filename
    if not path.exists():
        st.info(f"Résultat non disponible : {path.relative_to(path.parents[2])}")
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        st.warning(f"Impossible de lire {path.name} : {exc}")
        return None


def _result_json(directory: Path, filename: str) -> dict:
    path = directory / filename
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Impossible de lire {path.name} : {exc}")
        return {}


def _science_has(frame: pd.DataFrame | None, columns: list[str]) -> bool:
    if frame is None:
        return False
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        st.warning("Colonnes absentes du résultat : " + ", ".join(missing))
        return False
    return True


def _science_results_match_active_data(df: pd.DataFrame) -> bool:
    """Empêche d'afficher des métriques calculées sur une ancienne base."""
    metadata = _science_json("statistical_validation_metadata.json")
    expected_rows = metadata.get("dataset_rows")
    expected_hash = str(metadata.get("dataset_sha256", "")).strip().lower()
    data_path = find_data_file()
    current_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    if expected_rows is not None and int(expected_rows) != len(df):
        st.error(
            "Les résultats scientifiques ont été calculés sur "
            f"{int(expected_rows):,} lignes, mais la base active en contient "
            f"{len(df):,}. Relancez les expériences avant d'interpréter les "
            "tableaux."
        )
        return False
    if expected_hash and expected_hash != current_hash:
        st.error(
            "La base active a changé depuis la génération des résultats "
            "scientifiques (empreinte SHA-256 différente). Les anciens tableaux "
            "sont masqués pour éviter une conclusion périmée."
        )
        return False
    if not expected_hash:
        st.warning(
            "L'ancienne métadonnée ne contient pas d'empreinte SHA-256. Le nombre "
            "de lignes correspond, mais la traçabilité complète exige de relancer "
            "le protocole scientifique."
        )
    else:
        st.success(
            "Traçabilité vérifiée : les résultats scientifiques correspondent à "
            f"{data_path.name} (SHA-256 {current_hash[:12]}…)."
        )
    return True


def _render_science_cv() -> None:
    st.markdown("#### Validation croisée répétée 5 × 5")
    metadata = _science_json("statistical_validation_metadata.json")
    protocol = metadata.get("validation", {})
    summary = _science_csv("statistical_cv_summary_95ci.csv")
    required = [
        "model",
        "mae_normalized_mean",
        "mae_normalized_std",
        "rmse_normalized_mean",
        "rmse_normalized_std",
        "rmse_normalized_ci95_low",
        "rmse_normalized_ci95_high",
        "r2_macro_mean",
        "r2_macro_std",
        "r2_macro_ci95_low",
        "r2_macro_ci95_high",
    ]
    if not _science_has(summary, required):
        return

    summary = summary.copy()
    summary["Groupe"] = "Références statistiques"
    heavy_metadata = _science_json("heavy_validation_metadata.json")
    heavy_protocol = heavy_metadata.get("validation", {})
    heavy_is_complete = (
        int(heavy_protocol.get("splits", 0)) == 5
        and int(heavy_protocol.get("repeats", 0)) == 5
        and int(heavy_protocol.get("total_folds", 0)) == 25
    )
    if heavy_is_complete:
        heavy_summary = _science_csv("heavy_cv_summary_95ci.csv")
        if _science_has(heavy_summary, required):
            heavy_summary = heavy_summary.copy()
            heavy_summary["Groupe"] = "Modèles lourds et hybrides"
            summary = pd.concat([summary, heavy_summary], ignore_index=True)
    else:
        st.warning(
            "La validation 5×5 des modèles lourds n'est pas complète. "
            "Le tableau principal affiche donc uniquement les références "
            "statistiques validées sur 25 plis."
        )

    ranked = summary.sort_values("rmse_normalized_mean").reset_index(drop=True)
    best = ranked.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Plis exécutés",
        int(protocol.get("total_folds", protocol.get("splits", 0))),
    )
    c2.metric("Meilleur modèle CV", str(best["model"]))
    c3.metric("RMSE normalisé", f"{best['rmse_normalized_mean']:.3f}")
    c4.metric("R² macro", f"{best['r2_macro_mean']:.3f}")
    st.caption(
        f"{protocol.get('method', 'RepeatedKFold')} · "
        f"{protocol.get('splits', 5)} plis × {protocol.get('repeats', 5)} "
        "répétitions · mêmes partitions pour tous les modèles. Les intervalles "
        "à 95 % sont calculés entre répétitions."
    )
    if len(ranked) > 1:
        rmse_gap = float(ranked.loc[1, "rmse_normalized_mean"]) - float(
            ranked.loc[0, "rmse_normalized_mean"]
        )
        st.info(
            f"{ranked.loc[0, 'model']} est premier numériquement, mais l'écart "
            f"avec {ranked.loc[1, 'model']} n'est que de {rmse_gap:.6f} en RMSE "
            "normalisé et leurs IC95 se chevauchent. Ils doivent être considérés "
            "comme pratiquement ex æquo."
        )

    table = ranked[["Groupe"] + required].rename(
        columns={
            "Groupe": "Groupe",
            "model": "Modèle",
            "mae_normalized_mean": "MAE norm. moyenne",
            "mae_normalized_std": "MAE norm. écart-type",
            "rmse_normalized_mean": "RMSE norm. moyenne",
            "rmse_normalized_std": "RMSE norm. écart-type",
            "rmse_normalized_ci95_low": "RMSE IC95 bas",
            "rmse_normalized_ci95_high": "RMSE IC95 haut",
            "r2_macro_mean": "R² macro moyen",
            "r2_macro_std": "R² macro écart-type",
            "r2_macro_ci95_low": "R² IC95 bas",
            "r2_macro_ci95_high": "R² IC95 haut",
        }
    )
    st.dataframe(
        table.style.format(
            {
                column: "{:.4f}"
                for column in table.columns
                if column not in {"Groupe", "Modèle"}
            }
        ),
        width="stretch",
        hide_index=True,
    )

    figure = go.Figure(
        go.Bar(
            x=ranked["model"],
            y=ranked["r2_macro_mean"],
            marker_color="#2b7a67",
            error_y={
                "type": "data",
                "array": (
                    ranked["r2_macro_ci95_high"] - ranked["r2_macro_mean"]
                ).clip(lower=0),
                "arrayminus": (
                    ranked["r2_macro_mean"] - ranked["r2_macro_ci95_low"]
                ).clip(lower=0),
            },
        )
    )
    figure.add_hline(
        y=0,
        line_dash="dash",
        line_color="#b04b45",
        annotation_text="Prédicteur moyen : R² ≈ 0",
    )
    figure.update_layout(
        title="R² macro moyen et intervalle de confiance à 95 %",
        xaxis_title="Modèle",
        yaxis_title="R² macro",
        template="plotly_white",
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch")

    details = _science_csv("statistical_cv_by_output_95ci.csv")
    detail_required = [
        "model",
        "target",
        "mae_mean",
        "rmse_mean",
        "r2_mean",
        "r2_ci95_low",
        "r2_ci95_high",
        "rmse_improvement_vs_dummy_pct",
    ]
    if not _science_has(details, detail_required):
        return
    details = details.copy()
    details["Groupe"] = "Références statistiques"
    if heavy_is_complete:
        heavy_details = _science_csv("heavy_cv_by_output_95ci.csv")
        if _science_has(heavy_details, detail_required):
            heavy_details = heavy_details.copy()
            heavy_details["Groupe"] = "Modèles lourds et hybrides"
            dummy_rmse = (
                details[details["model"] == "Naive moyenne"]
                .set_index("target")["rmse_mean"]
                .to_dict()
            )
            missing_gain = heavy_details["rmse_improvement_vs_dummy_pct"].isna()
            heavy_details.loc[missing_gain, "rmse_improvement_vs_dummy_pct"] = (
                1.0
                - heavy_details.loc[missing_gain, "rmse_mean"]
                / heavy_details.loc[missing_gain, "target"].map(dummy_rmse)
            ) * 100.0
            details = pd.concat([details, heavy_details], ignore_index=True)
    target = st.selectbox(
        "Sortie analysée",
        TARGET_COLUMNS,
        format_func=label,
        key="science_cv_target",
    )
    rows = details[details["target"] == target].sort_values("rmse_mean")
    optional = [
        column
        for column in ("rmse_over_mean_pct_mean", "nrmse_range_pct_mean")
        if column in rows.columns
    ]
    detail_table = rows[["Groupe"] + detail_required + optional].rename(
        columns={
            "Groupe": "Groupe",
            "model": "Modèle",
            "target": "Sortie",
            "mae_mean": "MAE moyenne",
            "rmse_mean": "RMSE moyenne",
            "r2_mean": "R² moyen",
            "r2_ci95_low": "R² IC95 bas",
            "r2_ci95_high": "R² IC95 haut",
            "rmse_improvement_vs_dummy_pct": "Gain RMSE vs moyenne (%)",
            "rmse_over_mean_pct_mean": "RMSE / moyenne (%)",
            "nrmse_range_pct_mean": "RMSE / étendue (%)",
        }
    )
    st.dataframe(
        detail_table.style.format(
            {
                column: ("{:+.2f}" if "Gain" in column else "{:.4f}")
                for column in detail_table.columns
                if column not in {"Groupe", "Modèle", "Sortie"}
            }
        ),
        width="stretch",
        hide_index=True,
    )
    target_figure = go.Figure(
        go.Bar(
            x=rows["model"],
            y=rows["r2_mean"],
            marker_color="#4e7fa8",
            error_y={
                "type": "data",
                "array": (rows["r2_ci95_high"] - rows["r2_mean"]).clip(lower=0),
                "arrayminus": (
                    rows["r2_mean"] - rows["r2_ci95_low"]
                ).clip(lower=0),
            },
        )
    )
    target_figure.add_hline(y=0, line_dash="dash", line_color="#b04b45")
    target_figure.update_layout(
        title=f"R² par modèle — {label(target)}",
        xaxis_title="Modèle",
        yaxis_title="R²",
        template="plotly_white",
    )
    st.plotly_chart(target_figure, width="stretch")
    st.warning(
        "Le R² macro global est dominé par les gaz bien prédits. Les trois "
        "rendements doivent être interprétés séparément et comparés au prédicteur "
        "naïf par la moyenne."
    )
    if heavy_is_complete:
        st.caption(
            "Le « MLP Optuna fixe » utilise une architecture déjà choisie et "
            "l'évalue sur les 25 plis. La sélection d'hyperparamètres entièrement "
            "imbriquée est rapportée séparément dans l'onglet Complexité & Optuna."
        )


def _render_science_oof() -> None:
    st.markdown("#### Diagnostics hors pli — référence Ridge")
    oof = _science_csv("ridge_oof_predictions.csv")
    if oof is None:
        return
    target = st.selectbox(
        "Sortie pour les diagnostics OOF",
        TARGET_COLUMNS,
        format_func=label,
        key="science_oof_target",
    )
    observed = f"Observed__{target}"
    predicted = f"Predicted__{target}"
    residual = f"Residual__{target}"
    if not _science_has(oof, [observed, predicted, residual]):
        return
    st.caption(
        "Chaque observation est prédite par un modèle qui ne l’a pas utilisée "
        "pendant son ajustement."
    )
    left, right = st.columns(2)
    with left:
        low = float(min(oof[observed].min(), oof[predicted].min()))
        high = float(max(oof[observed].max(), oof[predicted].max()))
        figure = px.scatter(
            oof,
            x=observed,
            y=predicted,
            opacity=0.58,
            labels={
                observed: f"Observé — {label(target)}",
                predicted: f"Prédit hors pli — {label(target)}",
            },
            title="Observé versus prédit",
            template="plotly_white",
        )
        figure.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                name="Accord parfait",
                line={"dash": "dash", "color": "#b04b45"},
            )
        )
        st.plotly_chart(figure, width="stretch")
    with right:
        figure = px.histogram(
            oof,
            x=residual,
            nbins=36,
            marginal="box",
            labels={residual: f"Résidu observé − prédit ({label(target)})"},
            title="Distribution des résidus",
            template="plotly_white",
        )
        figure.add_vline(x=0, line_dash="dash", line_color="#b04b45")
        st.plotly_chart(figure, width="stretch")

    inputs = [column for column in INPUT_COLUMNS if column in oof.columns]
    if inputs:
        selected_input = st.selectbox(
            "Entrée pour rechercher un biais résiduel",
            inputs,
            format_func=label,
            key="science_oof_input",
        )
        figure = px.scatter(
            oof,
            x=selected_input,
            y=residual,
            opacity=0.55,
            labels={
                selected_input: label(selected_input),
                residual: f"Résidu — {label(target)}",
            },
            title="Résidus selon une condition opératoire",
            template="plotly_white",
        )
        figure.add_hline(y=0, line_dash="dash", line_color="#b04b45")
        st.plotly_chart(figure, width="stretch")

    raw = _science_csv("raw_vs_constrained_metrics.csv")
    mass = _science_csv("mass_balance_constraints.csv")
    if raw is not None or mass is not None:
        st.markdown("#### Effet des règles numériques de cohérence")
        left, right = st.columns(2)
        with left:
            if raw is not None:
                st.dataframe(raw, width="stretch", hide_index=True)
        with right:
            if mass is not None:
                st.dataframe(mass, width="stretch", hide_index=True)
        st.caption(
            "Les métriques brutes et contraintes sont séparées afin que la "
            "normalisation du bilan de masse ne masque pas l’erreur du modèle."
        )


def _render_science_learning() -> None:
    st.markdown("#### Courbe d’apprentissage")
    learning = _science_csv("ridge_learning_curve_summary.csv")
    if _science_has(
        learning,
        [
            "training_size",
            "split",
            "rmse_normalized_mean",
            "rmse_normalized_std",
            "r2_macro_mean",
            "r2_macro_std",
        ],
    ):
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.line(
                    learning,
                    x="training_size",
                    y="rmse_normalized_mean",
                    color="split",
                    markers=True,
                    error_y="rmse_normalized_std",
                    labels={
                        "training_size": "Observations d’entraînement",
                        "rmse_normalized_mean": "RMSE normalisé moyen",
                        "split": "Jeu",
                    },
                    title="Erreur selon la taille d’apprentissage",
                    template="plotly_white",
                ),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                px.line(
                    learning,
                    x="training_size",
                    y="r2_macro_mean",
                    color="split",
                    markers=True,
                    error_y="r2_macro_std",
                    labels={
                        "training_size": "Observations d’entraînement",
                        "r2_macro_mean": "R² macro moyen",
                        "split": "Jeu",
                    },
                    title="Variance expliquée selon la taille d’apprentissage",
                    template="plotly_white",
                ),
                width="stretch",
            )

    st.markdown("#### Ablation des variables dérivées")
    ablation = _science_csv("ridge_feature_ablation_summary.csv")
    if _science_has(
        ablation,
        [
            "feature_set",
            "rmse_normalized_mean",
            "rmse_normalized_std",
            "r2_macro_mean",
            "r2_macro_std",
        ],
    ):
        st.dataframe(
            ablation.style.format(
                {
                    column: "{:.4f}"
                    for column in ablation.columns
                    if column != "feature_set"
                }
            ),
            width="stretch",
            hide_index=True,
        )
        figure = px.bar(
            ablation,
            x="feature_set",
            y="r2_macro_mean",
            error_y="r2_macro_std",
            color="feature_set",
            labels={
                "feature_set": "Variables",
                "r2_macro_mean": "R² macro moyen",
            },
            title="5 entrées brutes versus 5 entrées + 10 variables dérivées",
            template="plotly_white",
        )
        figure.update_layout(showlegend=False)
        st.plotly_chart(figure, width="stretch")
        st.info(
            "Les 10 variables dérivées sont des transformations déterministes. "
            "Elles ne créent ni nouvelle expérience ni information physique mesurée."
        )

    vif = _science_csv("feature_multicollinearity_vif.csv")
    if vif is not None:
        with st.expander("Audit de multicolinéarité des 15 caractéristiques"):
            st.dataframe(vif, width="stretch", hide_index=True)
    augmentation = _science_csv("augmentation_summary.csv")
    if augmentation is not None:
        with st.expander("Augmentation appliquée uniquement au train"):
            st.dataframe(augmentation, width="stretch", hide_index=True)
            st.warning(
                "Bootstrap, jitter et Mixup sont des régularisations. Ils ne "
                "remplacent pas de nouvelles mesures expérimentales."
            )
    diagnostic_curve = (
        SCIENTIFIC_REPORT_DIR
        / "diagnostics"
        / "learning_curves_ridge_mlp_compact.png"
    )
    if diagnostic_curve.exists():
        with st.expander("Courbe comparative Ridge / MLP compact — exploratoire"):
            st.image(
                str(diagnostic_curve),
                caption=(
                    "Comparaison complémentaire sur découpage fixe; la courbe "
                    "Ridge répétée ci-dessus reste la preuve principale."
                ),
                width="stretch",
            )


def _render_science_complexity() -> None:
    st.markdown("#### Complexité des réseaux")
    complexity = None
    for filename in (
        "heavy_model_complexity.csv",
        "statistical_model_complexity.csv",
        "model_complexity.csv",
    ):
        if (SCIENTIFIC_REPORT_DIR / filename).exists():
            complexity = _science_csv(filename)
            break
    required = [
        "model",
        "architecture",
        "parameters",
        "training_observations",
        "observations_per_parameter",
        "required_observations_5x",
        "required_observations_10x",
        "rule_5_to_10_satisfied",
    ]
    if _science_has(complexity, required):
        st.dataframe(
            complexity.style.format(
                {
                    "parameters": "{:,.0f}",
                    "training_observations": "{:,.0f}",
                    "observations_per_parameter": "{:.3f}",
                    "required_observations_5x": "{:,.0f}",
                    "required_observations_10x": "{:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.plotly_chart(
            px.bar(
                complexity,
                x="model",
                y="parameters",
                color="rule_5_to_10_satisfied",
                log_y=True,
                labels={
                    "model": "MLP",
                    "parameters": "Paramètres — échelle logarithmique",
                    "rule_5_to_10_satisfied": "Règle 5–10 respectée",
                },
                title="Paramètres entraînables face aux 800 observations",
                template="plotly_white",
            ),
            width="stretch",
        )
        rule_values = complexity["rule_5_to_10_satisfied"]
        if rule_values.dtype == object:
            rule_values = (
                rule_values.astype(str).str.strip().str.casefold() == "true"
            )
        else:
            rule_values = rule_values.astype(bool)
        compatible = complexity[rule_values]
        incompatible = complexity[~rule_values]
        if not compatible.empty:
            st.success(
                "Architecture compatible avec le seuil minimal de 5 observations "
                "par paramètre : "
                + ", ".join(compatible["model"].astype(str))
                + "."
            )
        if not incompatible.empty:
            st.error(
                "Architectures incompatibles avec la règle indicative de 5 à 10 "
                "observations par paramètre : "
                + ", ".join(incompatible["model"].astype(str))
                + ". Le MLP compact reste la référence neuronale défendable "
                "avec la base actuelle."
            )

    st.markdown("#### Optuna avec validation imbriquée")
    nested = _science_csv("nested_optuna_summary.csv")
    if _science_has(
        nested,
        [
            "outer_folds",
            "inner_folds",
            "trials_per_outer_fold",
            "rmse_normalized_mean",
            "rmse_normalized_std",
            "r2_macro_mean",
            "r2_macro_std",
        ],
    ):
        row = nested.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Plis externes", int(row["outer_folds"]))
        c2.metric("Plis internes", int(row["inner_folds"]))
        c3.metric("Essais / pli", int(row["trials_per_outer_fold"]))
        c4.metric("R² externe moyen", f"{row['r2_macro_mean']:.3f}")
        st.caption(
            f"RMSE externe : {row['rmse_normalized_mean']:.4f} ± "
            f"{row['rmse_normalized_std']:.4f}. Le test final n’est jamais "
            "utilisé pour sélectionner les hyperparamètres."
        )
        st.info(
            "Cette étude Optuna est un calcul séparé de la validation répétée "
            f"5×5. Avec {int(row['trials_per_outer_fold'])} essais par pli, elle valide surtout le "
            "protocole imbriqué; elle ne constitue pas une recherche TPE "
            "approfondie."
        )

    outer = _science_csv("nested_optuna_outer_folds.csv")
    if _science_has(
        outer,
        [
            "outer_fold",
            "architecture",
            "parameters",
            "inner_best_rmse_normalized",
            "outer_rmse_normalized",
            "outer_r2_macro",
        ],
    ):
        st.dataframe(outer, width="stretch", hide_index=True)
        st.plotly_chart(
            px.line(
                outer,
                x="outer_fold",
                y=["inner_best_rmse_normalized", "outer_rmse_normalized"],
                markers=True,
                labels={
                    "outer_fold": "Pli externe",
                    "value": "RMSE normalisé",
                    "variable": "Évaluation",
                },
                title="Erreur interne de sélection et erreur externe",
                template="plotly_white",
            ),
            width="stretch",
        )
    per_output = _science_csv("nested_optuna_by_output_summary.csv")
    if per_output is not None:
        with st.expander("Métriques Optuna imbriquées par sortie"):
            st.dataframe(per_output, width="stretch", hide_index=True)
    st.warning(
        "Cette étude imbriquée distincte vérifie l'absence de fuite de sélection, "
        "mais elle ne remplace ni la campagne canonique renforcée ni une validation "
        "expérimentale externe."
    )


def _render_science_composition(df: pd.DataFrame) -> None:
    st.markdown("#### Rendements comme composition massique")
    composition = _science_csv("composition_yields_summary.csv")
    if _science_has(
        composition,
        [
            "method",
            "rmse_normalized_mean",
            "r2_macro_yields_mean",
            "mass_balance_error_mean",
            "minimum_predicted_yield",
        ],
    ):
        st.dataframe(
            composition.style.format(
                {
                    column: "{:.4f}"
                    for column in composition.columns
                    if column not in {"method", "repetitions"}
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.plotly_chart(
            px.bar(
                composition,
                x="method",
                y="r2_macro_yields_mean",
                error_y=(
                    "r2_macro_yields_std"
                    if "r2_macro_yields_std" in composition.columns
                    else None
                ),
                color="method",
                labels={
                    "method": "Méthode",
                    "r2_macro_yields_mean": "R² macro des rendements",
                },
                title="ALR compositionnel versus renormalisation directe",
                template="plotly_white",
            ),
            width="stretch",
        )
        st.caption(
            "L’ALR garantit des rendements positifs totalisant 100 %. Il "
            "n’améliore pas le Ridge direct dans l’expérience actuelle : la "
            "contrainte physique est respectée, mais le signal reste limité."
        )
    per_output = _science_csv("composition_yields_by_output_summary.csv")
    if per_output is not None:
        with st.expander("Résultats compositionnels par rendement"):
            st.dataframe(per_output, width="stretch", hide_index=True)

    st.markdown("#### Provenance et portée scientifique")
    sidecar_path = (
        Path(__file__).resolve().parent
        / "data_enriched"
        / "e_waste_pyrolysis_merged_1000_provenance.csv"
    )
    sidecar: pd.DataFrame | None = None
    if sidecar_path.exists():
        try:
            sidecar = pd.read_csv(sidecar_path, encoding="utf-8-sig")
        except Exception as exc:
            st.warning(f"Impossible de lire le sidecar de provenance : {exc}")
    expected = [
        "Record_ID",
        "Source_DOI",
        "Source_file",
        "Study_ID",
        "Batch_ID",
        "Merge_method",
        "Data_quality_flag",
        "Feedstock",
    ]
    lookup = {str(column).casefold(): str(column) for column in df.columns}
    sidecar_lookup = (
        {str(column).casefold(): str(column) for column in sidecar.columns}
        if sidecar is not None
        else {}
    )
    provenance = pd.DataFrame(
        [
            {
                "Métadonnée": column,
                "Disponible": (
                    column.casefold() in lookup
                    or column.casefold() in sidecar_lookup
                ),
                "Emplacement": (
                    "Base active"
                    if column.casefold() in lookup
                    else (
                        "Sidecar d'audit"
                        if column.casefold() in sidecar_lookup
                        else "Absente"
                    )
                ),
                "Colonne trouvée": lookup.get(
                    column.casefold(),
                    sidecar_lookup.get(column.casefold(), "—"),
                ),
            }
            for column in expected
        ]
    )
    st.dataframe(provenance, width="stretch", hide_index=True)
    present = int(provenance["Disponible"].sum())

    sidecar_required = [
        "Record_ID",
        "Mass_balance_pct",
        "Merge_domain_status",
        "Data_quality_flag",
        "Merge_method",
    ]
    if sidecar is not None and all(
        column in sidecar.columns for column in sidecar_required
    ):
        domain_counts = (
            sidecar["Merge_domain_status"]
            .value_counts(dropna=False)
            .rename_axis("Statut de domaine")
            .reset_index(name="Lignes")
        )
        domain_counts["Proportion (%)"] = (
            100.0 * domain_counts["Lignes"] / len(sidecar)
        )
        left, right = st.columns(2)
        with left:
            st.dataframe(domain_counts, width="stretch", hide_index=True)
        with right:
            st.metric("Identifiants uniques", sidecar["Record_ID"].nunique())
            st.metric(
                "Bilan massique observé",
                f"{sidecar['Mass_balance_pct'].min():.2f}–"
                f"{sidecar['Mass_balance_pct'].max():.2f} %",
            )
        st.warning(
            "Les 543 interpolations et 457 extrapolations sont des estimations "
            "synthétiques de prototype, pas 1 000 nouvelles mesures "
            "expérimentales indépendantes. Le sidecar est exclu des features."
        )
        st.download_button(
            "Télécharger le sidecar de provenance",
            sidecar.to_csv(index=False).encode("utf-8-sig"),
            file_name=sidecar_path.name,
            mime="text/csv",
            key="download_science_provenance",
        )

    if present == 0:
        st.error(
            "La base active ne contient aucun identifiant de source, d’étude, de "
            "lot ou de qualité. Ces métadonnées ne sont pas des entrées du modèle, "
            "mais elles doivent être conservées dans une table latérale pour "
            "auditer les lignes et effectuer une validation par étude."
        )
    elif present < len(expected):
        st.warning(
            "La provenance restaurée reste partielle : DOI, Study_ID, Batch_ID "
            "et Feedstock manquent encore. Ils sont nécessaires pour une "
            "validation GroupKFold ou leave-one-source-out."
        )

    metadata = _science_json("statistical_validation_metadata.json")
    var = metadata.get("var_applicability", {})
    st.info(
        "VAR non appliquée : "
        + str(
            var.get(
                "reason",
                "les expériences n’ont ni ordre temporel ni structure de retards.",
            )
        )
        + " Kernel Ridge RBF et SVR-RBF sont les références à noyau adaptées."
    )
    st.warning(
        "La validation 5×5 mesure la stabilité interne de cette base. Elle ne "
        "prouve pas la transférabilité à un autre réacteur, une autre étude ou "
        "une campagne industrielle."
    )


def _render_active_nested_validation(df: pd.DataFrame) -> None:
    """Présente la validation imbriquée des seules familles interactives."""
    st.markdown("#### Validation imbriquée répétée des familles actives")
    metadata = _result_json(NESTED_VALIDATION_DIR, "run_metadata.json")
    if not metadata:
        st.warning(
            "Le calcul canonique n’est pas encore disponible. Exécuter "
            "`experiments/run_nested_repeated_optuna_validation.py` avant "
            "d’interpréter cette section."
        )
        return

    dataset_meta = metadata.get("dataset", {})
    data_path = find_data_file()
    current_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    expected_hash = str(dataset_meta.get("sha256", "")).strip().lower()
    expected_rows = dataset_meta.get("rows_after_cleaning")
    if (
        expected_hash
        and expected_hash != current_hash
        or expected_rows is not None
        and int(expected_rows) != len(df)
    ):
        st.error(
            "Les résultats imbriqués ne correspondent pas à la base active. "
            "Ils sont masqués afin d’éviter une conclusion périmée."
        )
        return

    protocol = metadata.get("validation", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plis externes", int(protocol.get("outer_splits", 0)))
    c2.metric("Répétitions", int(protocol.get("repeats", 0)))
    c3.metric("Plis internes", int(protocol.get("inner_splits", 0)))
    c4.metric(
        "Essais / famille / cible",
        int(protocol.get("trials_per_family_target_outer_fold", 0)),
    )
    st.success(
        "Empreinte de la base vérifiée · mêmes plis externes pour le prédicteur "
        "naïf, Ridge polynomial Optuna, Extra Trees Optuna, MLP Optuna compact, "
        "l’hybride 2 branches MLP + XGBoost Optuna et la sélection automatique."
    )
    st.caption(
        "Les prétraitements, le feature engineering et Optuna sont ajustés "
        "exclusivement dans le train de chaque pli externe. Les intervalles t "
        "nominaux à 95 % utilisent la répétition OOF complète comme unité "
        "d’échantillonnage."
    )

    summary = _result_csv(NESTED_VALIDATION_DIR, "summary_global_95ci.csv")
    required_global = [
        "model",
        "mae_normalized_mean",
        "mae_normalized_std",
        "mae_normalized_ci95_low",
        "mae_normalized_ci95_high",
        "rmse_normalized_mean",
        "rmse_normalized_std",
        "rmse_normalized_ci95_low",
        "rmse_normalized_ci95_high",
        "r2_macro_mean",
        "r2_macro_std",
        "r2_macro_ci95_low",
        "r2_macro_ci95_high",
        "rmse_improvement_vs_naive_pct_mean",
    ]
    if not _science_has(summary, required_global):
        return
    ranked = summary.sort_values("rmse_normalized_mean").reset_index(drop=True)
    global_table = ranked[required_global].rename(
        columns={
            "model": "Modèle",
            "mae_normalized_mean": "MAE norm. moyenne",
            "mae_normalized_std": "MAE norm. écart-type",
            "mae_normalized_ci95_low": "MAE IC95 bas",
            "mae_normalized_ci95_high": "MAE IC95 haut",
            "rmse_normalized_mean": "RMSE norm. moyenne",
            "rmse_normalized_std": "RMSE norm. écart-type",
            "rmse_normalized_ci95_low": "RMSE IC95 bas",
            "rmse_normalized_ci95_high": "RMSE IC95 haut",
            "r2_macro_mean": "R² macro moyen",
            "r2_macro_std": "R² macro écart-type",
            "r2_macro_ci95_low": "R² IC95 bas",
            "r2_macro_ci95_high": "R² IC95 haut",
            "rmse_improvement_vs_naive_pct_mean": "Gain RMSE vs naïf (%)",
        }
    )
    st.dataframe(
        global_table.style.format(
            {
                column: ("{:+.2f}" if "Gain" in column else "{:.4f}")
                for column in global_table.columns
                if column != "Modèle"
            }
        ),
        width="stretch",
        hide_index=True,
    )

    figure = go.Figure(
        go.Bar(
            x=ranked["model"],
            y=ranked["r2_macro_mean"],
            marker_color="#2b7a67",
            error_y={
                "type": "data",
                "array": (
                    ranked["r2_macro_ci95_high"] - ranked["r2_macro_mean"]
                ).clip(lower=0),
                "arrayminus": (
                    ranked["r2_macro_mean"] - ranked["r2_macro_ci95_low"]
                ).clip(lower=0),
            },
        )
    )
    figure.add_hline(
        y=0,
        line_dash="dash",
        line_color="#b04b45",
        annotation_text="Niveau naïf : R² ≈ 0",
    )
    figure.update_layout(
        title="Familles actives — R² macro avec IC95",
        xaxis_title="Modèle",
        yaxis_title="R² macro",
        template="plotly_white",
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch")

    details = _result_csv(
        NESTED_VALIDATION_DIR,
        "summary_by_output_95ci.csv",
    )
    required_output = [
        "model",
        "target",
        "mae_mean",
        "mae_std",
        "mae_ci95_low",
        "mae_ci95_high",
        "rmse_mean",
        "rmse_std",
        "rmse_ci95_low",
        "rmse_ci95_high",
        "r2_mean",
        "r2_std",
        "r2_ci95_low",
        "r2_ci95_high",
        "rmse_improvement_vs_naive_pct_mean",
    ]
    if _science_has(details, required_output):
        target = st.selectbox(
            "Sortie détaillée — protocole imbriqué",
            TARGET_COLUMNS,
            format_func=label,
            key="active_nested_target",
        )
        rows = details[details["target"] == target].sort_values("rmse_mean")
        output_table = rows[required_output].rename(
            columns={
                "model": "Modèle",
                "target": "Sortie",
                "mae_mean": "MAE moyenne",
                "mae_std": "MAE écart-type",
                "mae_ci95_low": "MAE IC95 bas",
                "mae_ci95_high": "MAE IC95 haut",
                "rmse_mean": "RMSE moyenne",
                "rmse_std": "RMSE écart-type",
                "rmse_ci95_low": "RMSE IC95 bas",
                "rmse_ci95_high": "RMSE IC95 haut",
                "r2_mean": "R² moyen",
                "r2_std": "R² écart-type",
                "r2_ci95_low": "R² IC95 bas",
                "r2_ci95_high": "R² IC95 haut",
                "rmse_improvement_vs_naive_pct_mean": "Gain RMSE vs naïf (%)",
            }
        )
        st.dataframe(
            output_table.style.format(
                {
                    column: ("{:+.2f}" if "Gain" in column else "{:.4f}")
                    for column in output_table.columns
                    if column not in {"Modèle", "Sortie"}
                }
            ),
            width="stretch",
            hide_index=True,
        )

    left, right = st.columns(2)
    with left:
        st.markdown("##### Stabilité de la sélection par sortie")
        frequency = _result_csv(
            NESTED_VALIDATION_DIR,
            "best_family_frequency_by_output.csv",
        )
        if frequency is not None:
            display_frequency = frequency.copy()
            display_frequency["target"] = display_frequency["target"].map(label)
            st.dataframe(
                display_frequency,
                width="stretch",
                hide_index=True,
            )
    with right:
        st.markdown("##### Complexité maximale des branches MLP réduites")
        complexity = _result_csv(
            NESTED_VALIDATION_DIR,
            "mlp_complexity.csv",
        )
        if complexity is not None and not complexity.empty:
            group_columns = [
                column
                for column in ("family", "mlp_role", "target")
                if column in complexity.columns
            ]
            complexity_summary = (
                complexity.groupby(group_columns, as_index=False)
                .agg(
                    paramètres_max=("parameter_count", "max"),
                    ratio_externe_min=(
                        "outer_observations_per_parameter",
                        "min",
                    ),
                    ratio_interne_min=(
                        "inner_observations_per_parameter",
                        "min",
                    ),
                    règle_5x_externe=(
                        "outer_rule_5x_satisfied",
                        "all",
                    ),
                    règle_5x_interne=(
                        "inner_rule_5x_satisfied",
                        "all",
                    ),
                )
            )
            complexity_summary["target"] = complexity_summary["target"].map(
                label
            )
            st.dataframe(
                complexity_summary.style.format(
                    {
                        "ratio_externe_min": "{:.2f}",
                        "ratio_interne_min": "{:.2f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    uncertainty = metadata.get("uncertainty", {})
    limitations = metadata.get("limitations", [])
    st.warning(
        "Interprétation : ces intervalles décrivent la sensibilité aux "
        f"{int(protocol.get('repeats', 0))} répétitions de partitions. "
        "Ils ne mesurent ni l’incertitude instrumentale ni la transférabilité "
        "à un autre réacteur. "
        + str(uncertainty.get("fold_level_intervals", ""))
    )
    if limitations:
        with st.expander("Limites scientifiques enregistrées avec le calcul"):
            for item in limitations:
                st.markdown(f"- {item}")


def _render_new_scientific_evidence() -> None:
    st.markdown("#### Nouvelles preuves graphiques")
    if not OPTUNA_EVIDENCE_DIR.exists():
        st.warning(
            "Les graphiques seront disponibles après l’agrégation de la "
            "validation imbriquée."
        )
        return
    figures = [
        ("01_model_comparison_r2_ic95.png", "Comparaison globale avec IC95"),
        ("02_r2_by_output_ic95.png", "R² par sortie avec IC95"),
        ("03_rmse_by_output_ic95.png", "RMSE par sortie avec IC95"),
        ("04_mae_by_output_ic95.png", "MAE par sortie avec IC95"),
        (
            "05_observed_vs_predicted_8_outputs.png",
            "Observé versus prédit hors pli — huit sorties",
        ),
        (
            "06_residual_distributions_8_outputs.png",
            "Distribution des résidus — huit sorties",
        ),
        (
            "07_residuals_vs_temperature.png",
            "Résidus selon la température",
        ),
        (
            "08_residuals_vs_concentration.png",
            "Résidus selon la concentration",
        ),
        (
            "09_oaat_sensitivity_8_outputs.png",
            "Sensibilité un facteur à la fois",
        ),
        (
            "10_scenarios_and_pareto_front.png",
            "Échantillonnage de scénarios et front de Pareto",
        ),
        ("11_learning_curve.png", "Courbe d’apprentissage"),
    ]
    available = [
        (OPTUNA_EVIDENCE_DIR / filename, title)
        for filename, title in figures
        if (OPTUNA_EVIDENCE_DIR / filename).exists()
    ]
    if not available:
        st.info("Aucune figure validée n’est encore disponible.")
        return
    columns = st.columns(2)
    for index, (path, title) in enumerate(available):
        with columns[index % 2]:
            st.image(str(path), caption=title, width="stretch")
    st.caption(
        "Les analyses de sensibilité et de Pareto décrivent le comportement du "
        "modèle figé dans le domaine observé. Elles ne démontrent pas une "
        "relation causale et ne constituent pas une consigne industrielle."
    )


def _render_preprocessing_traceability(df: pd.DataFrame) -> None:
    st.markdown("#### Prétraitement, variables et contraintes")
    duplicate_count = int(df.duplicated().sum())
    missing_inputs = int(df[INPUT_COLUMNS].isna().sum().sum())
    missing_targets = int(df[TARGET_COLUMNS].isna().sum().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes actives", len(df))
    c2.metric("Doublons exacts", duplicate_count)
    c3.metric("Valeurs manquantes — entrées", missing_inputs)
    c4.metric("Valeurs manquantes — sorties", missing_targets)

    st.caption(
        "Les doublons sont supprimés au chargement. Les cibles manquantes sont "
        "exclues ; les entrées manquantes sont imputées par la médiane à "
        "l’intérieur de chaque pipeline et donc de chaque pli."
    )
    ranges = pd.DataFrame(
        [
            {
                "Entrée": label(column),
                "Variable": column,
                "Minimum observé": float(df[column].min()),
                "Maximum observé": float(df[column].max()),
                "Unité": (
                    "kg"
                    if "kg" in column
                    else "°C"
                    if column == "Temperature_C"
                    else "min"
                    if column == "Temps_reaction_min"
                    else "°C/min"
                    if column == "Heating_rate_C_min"
                    else "mm"
                ),
            }
            for column in INPUT_COLUMNS
        ]
    )
    st.dataframe(
        ranges.style.format(
            {
                "Minimum observé": "{:.3f}",
                "Maximum observé": "{:.3f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("##### Feature engineering sans fuite de cible")
    feature_table = pd.DataFrame(
        [
            ("Thermal_Severity", "Température × temps de réaction"),
            ("Heating_Exposure", "Vitesse de chauffage × temps de réaction"),
            ("Temperature_Squared", "Température²"),
            (
                "Temperature_Heating_Interaction",
                "Température × vitesse de chauffage",
            ),
            (
                "Temperature_Particle_Ratio",
                "Température / (taille des particules + ε)",
            ),
            ("Log_Reaction_Time", "ln(1 + temps de réaction)"),
            (
                "Concentration_Particle_Ratio",
                "Quantité / (taille des particules + ε)",
            ),
            (
                "Thermal_Material_Load",
                "Quantité × température",
            ),
            (
                "Concentration_Time_Interaction",
                "Quantité × temps de réaction",
            ),
            (
                "Concentration_Heating_Interaction",
                "Quantité × vitesse de chauffage",
            ),
        ],
        columns=["Variable calculée", "Formule"],
    )
    st.dataframe(feature_table, width="stretch", hide_index=True)
    st.info(
        "Les ratios, conversions, sélectivités et charges halogénées calculés "
        "à partir des sorties sont réservés à l’analyse après prédiction. Ils ne "
        "sont jamais utilisés comme entrées des huit régressions."
    )

    st.markdown("##### Post-traitement numérique de cohérence")
    st.dataframe(
        pd.DataFrame(
            [
                (
                    "Valeurs négatives",
                    "Ramenées à zéro après assemblage des huit sorties",
                ),
                (
                    "Rendements",
                    "Bornés entre 0 et 100 %",
                ),
                (
                    "Bilan massique",
                    "Solid + Liquid + Gas renormalisés à 100 %",
                ),
                (
                    "Évaluation scientifique",
                    "Métriques imbriquées calculées sur les prédictions brutes",
                ),
            ],
            columns=["Règle", "Application"],
        ),
        width="stretch",
        hide_index=True,
    )
    st.warning(
        "Le post-traitement garantit la cohérence numérique des scénarios, mais "
        "ne doit pas masquer l’erreur brute du modèle. C’est pourquoi les "
        "preuves scientifiques sont calculées avant ces corrections."
    )


def render_scientific_validation(df: pd.DataFrame, bundle) -> None:
    del bundle
    st.subheader("Validation scientifique")
    st.markdown(
        '<div class="caption-card">Analyses exécutées pour répondre aux remarques '
        "du professeur : régression multi-sorties, prédicteur naïf, validation "
        "répétée et imbriquée, incertitude par sortie, diagnostics, sensibilité "
        "et hypothèses physiques.</div>",
        unsafe_allow_html=True,
    )
    if not NESTED_VALIDATION_DIR.exists():
        st.warning(
            "Le dossier de validation imbriquée n’est pas encore disponible. "
            "La page reste accessible, mais le calcul doit être généré."
        )
    report_files = [
        (
            "Annexe imbriquée active",
            NESTED_VALIDATION_DIR
            / "validation_imbriquee_optuna_professeur.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
        (
            "Rapport final des remarques",
            Path(__file__).resolve().parent
            / "reports"
            / "RAPPORT_FINAL_REMARQUES_PROFESSEUR.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (
            "Rapport final PDF",
            Path(__file__).resolve().parent
            / "reports"
            / "RAPPORT_FINAL_REMARQUES_PROFESSEUR.pdf",
            "application/pdf",
        ),
    ]
    available_reports = [
        item for item in report_files if item[1].exists()
    ]
    if available_reports:
        st.markdown("#### Livrables vérifiables")
        columns = st.columns(len(available_reports))
        for index, (title, path, mime) in enumerate(available_reports):
            columns[index].download_button(
                f"Télécharger — {title}",
                data=path.read_bytes(),
                file_name=path.name,
                mime=mime,
                key=f"science_report_{index}",
                width="stretch",
            )
    tabs = st.tabs(
        [
            "Familles actives — imbriquée",
            "Preuves nouvelles",
            "Prétraitement & traçabilité",
        ]
    )
    with tabs[0]:
        _render_active_nested_validation(df)
    with tabs[1]:
        _render_new_scientific_evidence()
    with tabs[2]:
        _render_preprocessing_traceability(df)


def render_optimization(bundle) -> None:
    st.subheader("Optimisation multicritère avec Optuna")
    st.caption(
        "Le score combine rendements, toxicité, quantité/coût du liquide, temps et "
        "coût énergétique. Toutes les recherches restent dans les plages observées."
    )
    with st.form("optimization_form"):
        c1, c2, c3 = st.columns(3)
        trials = c1.slider("Nombre d’essais Optuna", 50, 300, 150, 25)
        power = c2.number_input("Puissance du réacteur (kW)", 0.1, 1000.0, 10.0, 0.5)
        electricity = c3.number_input("Prix de l’électricité ($/kWh)", 0.0, 10.0, 0.15, 0.01)
        submitted = st.form_submit_button("Chercher la meilleure recette", type="primary")

    with st.expander("Pondérations du score final"):
        weights_table = pd.DataFrame(
            {
                "Objectif": [label(key) for key in DEFAULT_WEIGHTS],
                "Sens": ["Maximiser", "Maximiser", "Maximiser", "Minimiser", "Minimiser", "Minimiser", "Minimiser", "Minimiser"],
                "Poids": list(DEFAULT_WEIGHTS.values()),
            }
        )
        st.dataframe(weights_table, width="stretch", hide_index=True)
        st.caption(
            "Le coût énergétique suppose une puissance constante pendant la chauffe "
            "et la réaction. Cette hypothèse doit être remplacée par des mesures usine pour un coût réel."
        )

    if not submitted:
        st.info("Configurez la recherche puis lancez l’optimisation.")
        return

    assumptions = EnergyAssumptions(
        reactor_power_kw=float(power),
        electricity_price_usd_kwh=float(electricity),
    )
    with st.spinner(f"Exploration de {trials} recettes…"):
        result = optimize_recipe(
            bundle,
            n_trials=trials,
            assumptions=assumptions,
        )

    row = result.recommendation.iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Score optimal", f"{row['Score_final']:.1f} / 100")
    b.metric("Rendement liquide", f"{row['Liquid_Yield_pct']:.2f} %")
    c.metric("Rendement gaz", f"{row['Gas_Yield_pct']:.2f} %")
    d.metric("Toxicité", f"{row['Toxicity_Index_pct']:.3f} %")

    st.markdown("#### Proposition exploratoire")
    recommended_columns = INPUT_COLUMNS + [
        "Solid_Yield_pct",
        "Liquid_Yield_pct",
        "Gas_Yield_pct",
        "Toxicity_Index_pct",
        "Cout_liquide_USD",
        "Cout_energie_USD",
        "Score_final",
    ]
    display = result.recommendation[recommended_columns].rename(columns=DISPLAY_NAMES)
    st.dataframe(display.style.format(precision=3), width="stretch", hide_index=True)

    chart = px.line(
        result.history,
        x="Essai",
        y="Meilleur score cumulé",
        template="plotly_white",
        title="Convergence de l’optimisation",
        color_discrete_sequence=["#2b7a67"],
    )
    st.plotly_chart(chart, width="stretch")
    st.download_button(
        "Télécharger la proposition exploratoire (CSV)",
        display.to_csv(index=False).encode("utf-8-sig"),
        file_name="proposition_exploratoire_pyrowaste.csv",
        mime="text/csv",
    )


def main() -> None:
    data_path = find_data_file()
    df = get_data(str(data_path), data_path.stat().st_mtime)
    with st.spinner(
        "Chargement des quatre familles Optuna et de la sélection par sortie…"
    ):
        bundle = get_models(str(data_path), data_path.stat().st_mtime)
        strategy_assets = get_strategy_assets()
        strategy_summary, strategy_details = get_strategy_metrics(
            data_path.stat().st_mtime,
            ARTIFACT_PATH.stat().st_mtime,
        )

    with st.sidebar:
        st.markdown("## ♻️ PyroWaste")
        page = st.radio(
            "Navigation",
            [
                "Vue d’ensemble",
                "Comparaison modèles",
                "Validation scientifique",
                "Prédiction & optimisation",
                "SHAP",
            ],
        )
        st.divider()
        st.caption(f"Base : {Path(data_path).name}")
        st.caption("Séparation : 80 % / 10 % / 10 %")
        st.caption(
            f"{len(MODEL_STRATEGIES)} familles Optuna + 1 sélection automatique "
            f"= {len(ALL_STRATEGIES)} stratégies"
        )
        st.info(f"Présélection interne : {RECOMMENDED_STRATEGY}")

    hero()
    if page == "Vue d’ensemble":
        render_overview(df, bundle)
    elif page == "Comparaison modèles":
        render_models(strategy_summary, strategy_details, df, bundle)
    elif page == "Validation scientifique":
        render_scientific_validation(df, bundle)
    elif page == "Prédiction & optimisation":
        render_prediction(df, bundle, strategy_assets)
    else:
        render_shap(bundle)


if __name__ == "__main__":
    main()
