"""
Proyecto 2 – Saber 11 Valle del Cauca
Tablero interactivo de modelos predictivos
Grupo: La Tupla
"""

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

import tensorflow as tf

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN Y CARGA DE ARTEFACTOS
# ═══════════════════════════════════════════════════════════════════════════════

ARTIFACT_DIR = Path("artifacts_regresion")

# Modelos serializados (regresión - Jerónimo)
modelo_nn = tf.keras.models.load_model(

    ARTIFACT_DIR / "modelo_regresion_nn_saber11.keras"

)

modelo_hgb = joblib.load(

    ARTIFACT_DIR / "modelo_regresion_hgb_saber11.pkl"

)

scaler_X = joblib.load(ARTIFACT_DIR / "scaler_X.pkl")

scaler_y = joblib.load(ARTIFACT_DIR / "scaler_y.pkl")

with open(ARTIFACT_DIR / "metadata_regresion.json", encoding="utf-8") as f:
    metadata = json.load(f)

ORDINAL_MAPS = joblib.load(ARTIFACT_DIR / "ordinal_maps.pkl")
FEATURES     = metadata["features"]

# Tabla de comparación de experimentos
df_comp = pd.read_csv(ARTIFACT_DIR / "comparacion_final_regresion.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# OPCIONES LEGIBLES PARA DROPDOWNS
# ═══════════════════════════════════════════════════════════════════════════════

DROPDOWN_OPTIONS = {
    "FAMI_ESTRATOVIVIENDA": [
        "Sin Estrato", "Estrato 1", "Estrato 2", "Estrato 3",
        "Estrato 4", "Estrato 5", "Estrato 6"
    ],
    "FAMI_EDUCACIONPADRE": [
        "Ninguno", "No sabe", "Primaria incompleta", "Primaria completa",
        "Secundaria (Bachillerato) incompleta", "Secundaria (Bachillerato) completa",
        "Técnica o tecnológica incompleta", "Técnica o tecnológica completa",
        "Educación profesional incompleta", "Educación profesional completa",
        "Postgrado"
    ],
    "FAMI_EDUCACIONMADRE": [
        "Ninguno", "No sabe", "Primaria incompleta", "Primaria completa",
        "Secundaria (Bachillerato) incompleta", "Secundaria (Bachillerato) completa",
        "Técnica o tecnológica incompleta", "Técnica o tecnológica completa",
        "Educación profesional incompleta", "Educación profesional completa",
        "Postgrado"
    ],
    "FAMI_TIENEINTERNET":   ["No", "Si"],
    "FAMI_TIENECOMPUTADOR": ["No", "Si"],
    "FAMI_TIENELAVADORA":   ["No", "Si"],
    "COLE_BILINGUE":        ["N", "S"],
    "COLE_JORNADA":         ["SABATINA", "NOCHE", "TARDE", "MAÑANA", "UNICA", "COMPLETA"],
    "COLE_NATURALEZA":      ["OFICIAL", "NO OFICIAL"],
    "COLE_AREA_UBICACION":  ["RURAL", "URBANO"],
}

LABEL_MAP = {
    "FAMI_ESTRATOVIVIENDA": "Estrato de vivienda",
    "FAMI_EDUCACIONPADRE":  "Educación del padre",
    "FAMI_EDUCACIONMADRE":  "Educación de la madre",
    "FAMI_TIENEINTERNET":   "¿Tiene internet?",
    "FAMI_TIENECOMPUTADOR": "¿Tiene computador?",
    "FAMI_TIENELAVADORA":   "¿Tiene lavadora?",
    "COLE_BILINGUE":        "¿Colegio bilingüe?",
    "COLE_JORNADA":         "Jornada del colegio",
    "COLE_NATURALEZA":      "Naturaleza del colegio",
    "COLE_AREA_UBICACION":  "Área del colegio",
}

DEFAULT_VALUES = {
    "FAMI_ESTRATOVIVIENDA": "Estrato 2",
    "FAMI_EDUCACIONPADRE":  "Secundaria (Bachillerato) completa",
    "FAMI_EDUCACIONMADRE":  "Secundaria (Bachillerato) completa",
    "FAMI_TIENEINTERNET":   "Si",
    "FAMI_TIENECOMPUTADOR": "Si",
    "FAMI_TIENELAVADORA":   "Si",
    "COLE_BILINGUE":        "N",
    "COLE_JORNADA":         "MAÑANA",
    "COLE_NATURALEZA":      "OFICIAL",
    "COLE_AREA_UBICACION":  "URBANO",
}

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE PREDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_input(input_dict):

    """Convierte valores categóricos a numéricos usando los mapas ordinales."""

    row = []

    for feat in FEATURES:

        val = input_dict.get(feat)

        if val not in ORDINAL_MAPS[feat]:

            raise ValueError(

                f"Valor inválido para {feat}: {val}. "

                f"Valores válidos: {list(ORDINAL_MAPS[feat].keys())}"

            )

        mapped = ORDINAL_MAPS[feat][val]

        row.append(mapped)

    return np.array([row], dtype="float32")

def predecir_nn(input_dict):

    """Predice PUNT_GLOBAL usando la red neuronal."""

    X = _encode_input(input_dict)

    X_sc = scaler_X.transform(X).astype("float32")

    pred_sc = modelo_nn.predict(X_sc, verbose=0).ravel()[0]

    pred = scaler_y.inverse_transform(

        np.array([[pred_sc]], dtype="float32")

    ).ravel()[0]

    return float(pred)

def predecir_hgb(input_dict):

    """Predice PUNT_GLOBAL usando HistGradientBoostingRegressor."""

    X = _encode_input(input_dict)

    X_sc = scaler_X.transform(X).astype("float32")

    pred = modelo_hgb.predict(X_sc)[0]

    return float(pred)



# ═══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS ESTÁTICAS DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def fig_comparacion_modelos():
    """Barras agrupadas comparando MAE, RMSE y R² de todos los experimentos."""
    df = df_comp.copy()
    df["label"] = df["experimento"].str.replace("_", " ").str.title()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MAE", x=df["label"], y=df["MAE"],
        marker_color="#2c5f8a", text=df["MAE"].round(2), textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="RMSE", x=df["label"], y=df["RMSE"],
        marker_color="#c0392b", text=df["RMSE"].round(2), textposition="outside"
    ))
    fig.update_layout(
        barmode="group",
        title="Comparación de modelos",
        yaxis_title="Valor",
        template="plotly_white",
        height=400,
        margin=dict(t=60, b=40),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    return fig


def fig_r2_modelos():
    """Barras de R² para cada modelo."""
    df = df_comp.copy()
    df["label"] = df["experimento"].str.replace("_", " ").str.title()
    colores = ["#27ae60" if i == 0 else "#aec6cf" for i in range(len(df))]

    fig = go.Figure(go.Bar(
        x=df["label"], y=df["R2"],
        marker_color=colores,
        text=df["R2"].round(4), textposition="outside"
    ))
    fig.update_layout(
        title="R^2 de los modelos",
        yaxis_title="R^2",
        template="plotly_white",
        height=350,
        margin=dict(t=60, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# APP DASH
# ═══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Saber 11 — Modelos Predictivos",
)
server = app.server  # Para gunicorn en Docker

# ── Estilo global ─────────────────────────────────────────────────────────────
HEADER_STYLE = {
    "backgroundColor": "#1a2634",
    "color": "white",
    "padding": "1.2rem 2rem",
    "marginBottom": "0",
}

TAB_STYLE = {
    "padding": "6px 18px",
    "fontWeight": "500",
    "fontSize": "0.95rem",
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    "borderTop": "3px solid #2c5f8a",
    "fontWeight": "700",
}

# ── Header ────────────────────────────────────────────────────────────────────
header = html.Div([
    html.H2("Saber 11 — Valle del Cauca", style={"margin": 0, "fontWeight": "700"}),
    html.P("La Tupla - Jerónimo Rueda y JuanCamilo Gómez",
           style={"margin": 0, "opacity": 0.7, "fontSize": "0.9rem"}),
], style=HEADER_STYLE)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = dcc.Tabs(
    id="tabs-main",
    value="tab-regresion",
    children=[
        dcc.Tab(
            label="Predicción de Puntaje (Regresión)",
            value="tab-regresion",
            style=TAB_STYLE,
            selected_style=TAB_SELECTED_STYLE,
        ),
        dcc.Tab(
            label="Clasificación de Desempeño",
            value="tab-clasificacion",
            style=TAB_STYLE,
            selected_style=TAB_SELECTED_STYLE,
        ),
    ],
    style={"marginBottom": "0"},
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REGRESIÓN  (Jerónimo)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_dropdown(feature_name):
    """Genera un dropdown con label para una feature."""
    return html.Div([
        html.Label(
            LABEL_MAP[feature_name],
            style={"fontWeight": "600", "fontSize": "0.82rem", "marginBottom": "2px"},
        ),
        dcc.Dropdown(
            id=f"dd-{feature_name}",
            options=[{"label": v, "value": v} for v in DROPDOWN_OPTIONS[feature_name]],
            value=DEFAULT_VALUES[feature_name],
            clearable=False,
            style={"fontSize": "0.85rem"},
        ),
    ], style={"marginBottom": "10px"})


panel_inputs = dbc.Card([
    dbc.CardHeader(
        html.H5("Perfil del estudiante", className="mb-0", style={"fontWeight": "700"}),
    ),
    dbc.CardBody([

        html.H6("Condiciones del hogar",
                 style={"fontWeight": "700", "marginBottom": "8px", "color": "#2c5f8a"}),
        _make_dropdown("FAMI_ESTRATOVIVIENDA"),
        _make_dropdown("FAMI_EDUCACIONPADRE"),
        _make_dropdown("FAMI_EDUCACIONMADRE"),

        dbc.Row([
            dbc.Col(_make_dropdown("FAMI_TIENEINTERNET"),   width=4),
            dbc.Col(_make_dropdown("FAMI_TIENECOMPUTADOR"), width=4),
            dbc.Col(_make_dropdown("FAMI_TIENELAVADORA"),   width=4),
        ], className="mb-2"),

        html.Hr(),
        html.H6("Características del colegio",
                 style={"fontWeight": "700", "marginBottom": "8px", "color": "#2c5f8a"}),
        dbc.Row([
            dbc.Col(_make_dropdown("COLE_BILINGUE"),       width=6),
            dbc.Col(_make_dropdown("COLE_JORNADA"),        width=6),
        ]),
        dbc.Row([
            dbc.Col(_make_dropdown("COLE_NATURALEZA"),     width=6),
            dbc.Col(_make_dropdown("COLE_AREA_UBICACION"), width=6),
        ]),

        html.Div(
            dbc.Button(
                "Predecir puntaje", id="btn-predecir", color="primary",
                className="mt-3 w-100", size="lg",
            ),
        ),
    ]),
], className="shadow-sm")


panel_resultado = dbc.Card([
    dbc.CardHeader(
        html.H5("Resultado de la predicción", className="mb-0", style={"fontWeight": "700"}),
    ),
    dbc.CardBody(id="resultado-prediccion", children=[
        html.Div([
            html.P(
                'Ajuste el perfil del estudiante y presione "Predecir puntaje".',
                className="text-muted text-center mt-4",
                style={"fontSize": "1rem"},
            ),
        ])
    ]),
], className="shadow-sm")


panel_modelos = dbc.Card([
    dbc.CardHeader(
        html.H5("Desempeño de los modelos", className="mb-0", style={"fontWeight": "700"}),
    ),
    dbc.CardBody([
        dcc.Graph(id="graph-comparacion", figure=fig_comparacion_modelos(),
                  config={"displayModeBar": False}),
        dcc.Graph(id="graph-r2", figure=fig_r2_modelos(),
                  config={"displayModeBar": False}),
    ]),
], className="shadow-sm mt-3")


tab_regresion = html.Div([
    # Contexto de la pregunta
    dbc.Alert([
        "¿Pueden las condiciones socioeconómicas del hogar y las características del colegio "
        "predecir el puntaje global de un estudiante en el Saber 11?",
        html.Br(),
    ], color="info", className="mt-3 mb-3"),

    dbc.Row([
        # Columna izquierda: inputs
        dbc.Col(panel_inputs, lg=4, md=5, className="mb-3"),
        # Columna derecha: resultado + modelos
        dbc.Col([
            panel_resultado,
            panel_modelos,
        ], lg=8, md=7),
    ]),
], className="p-3")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLASIFICACIÓN  
# ═══════════════════════════════════════════════════════════════════════════════

tab_clasificacion = html.Div([
    dbc.Alert([,
        html.Br(),
    ], color="warning", className="mt-3 mb-3"),

    dbc.Card([
        dbc.CardBody([
            html.H4("Pestaña en construcción", className="text-center text-muted mt-5"),
            html.P(
                "Aquí irá el modelo de clasificación de Juan Camilo.",
                className="text-center text-muted mb-5",
            ),
        ])
    ], className="shadow-sm", style={"minHeight": "400px"}),
], className="p-3")


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

app.layout = html.Div([
    header,
    dbc.Container([
        tabs,
        html.Div(id="tab-content"),
    ], fluid=True),
])


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("tab-content", "children"),
    Input("tabs-main", "value"),
)
def render_tab(tab):
    if tab == "tab-regresion":
        return tab_regresion
    return tab_clasificacion


@callback(
    Output("resultado-prediccion", "children"),
    Input("btn-predecir", "n_clicks"),
    [State(f"dd-{f}", "value") for f in FEATURES],
    prevent_initial_call=True,
)
def hacer_prediccion(n_clicks, *values):
    try:
        input_dict = dict(zip(FEATURES, values))

        pred_nn = predecir_nn(input_dict)
        pred_hgb = predecir_hgb(input_dict)
        pred_final = pred_hgb

        fig_gauge = go.Figure()
        fig_gauge.add_trace(go.Bar(

                x=[pred_final],

                y=["Puntaje"],

                orientation="h",

                text=[f"{pred_final:.1f} pts"],

                textposition="outside",

                marker=dict(color="#2c5f8a"),

                hovertemplate="Puntaje predicho: %{x:.1f} pts<extra></extra>",

        ))

        # Línea vertical de referencia: media poblacional

        fig_gauge.add_vline(

            x=257,

            line_width=2,

            line_dash="dash",

            line_color="#c0392b",

            annotation_text="Media 257",

            annotation_position="top",

        )

        fig_gauge.update_layout(

            title={

                "text": "Puntaje Global Predicho",

                "x": 0.5,

                "xanchor": "center",

                "font": {"size": 18},

            },

            xaxis=dict(

                range=[0, 500],

                title="Puntaje Saber 11",

                tickmode="array",

                tickvals=[0, 100, 200, 300, 400, 500],

            ),

            yaxis=dict(

                title="",

                showticklabels=False,

            ),

            height=230,

            template="plotly_white",

            margin=dict(t=60, b=40, l=30, r=60),

            showlegend=False,

        )
                

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=["Red Neuronal", "Hist. Gradient Boosting"],
            y=[pred_nn, pred_hgb],
            marker_color=["#3498db", "#27ae60"],
            text=[f"{pred_nn:.1f}", f"{pred_hgb:.1f}"],
            textposition="outside",
            textfont={"size": 14, "color": "#1a2634"},
        ))

        fig_comp.update_layout(
            title="Predicción por modelo",
            yaxis_title="Puntaje predicho",
            yaxis_range=[0, max(pred_nn, pred_hgb) * 1.25],
            template="plotly_white",
            height=280,
            margin=dict(t=50, b=20),
        )


        return html.Div([
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=fig_gauge, config={"displayModeBar": False}),
                    lg=6,
                ),
                dbc.Col(
                    dcc.Graph(figure=fig_comp, config={"displayModeBar": False}),
                    lg=6,
                ),
            ]),

            

        ])

    except Exception as e:
        return dbc.Alert(
            [
                html.Strong("Error al generar la predicción: "),
                html.Code(str(e)),
            ],
            color="danger",
            className="mt-2",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
