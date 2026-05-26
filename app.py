"""
Proyecto 2
La Tupla
Jerónimo rueda - 202223775
Juan Camilo Gómez - 202220238
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


# CARGA DE ARTEFACTOS


BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "artifacts_regresion"
ARTIFACT_CLF_DIR = BASE_DIR / "artifacts_clasificacion"
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

df_comp = pd.read_csv(ARTIFACT_DIR / "comparacion_final_regresion.csv")


# Artefactos clasificación




modelo_clf_nn = tf.keras.models.load_model(
    ARTIFACT_CLF_DIR / "modelo_clasificacion_nn.keras"
)

modelo_clf_rf = joblib.load(
    ARTIFACT_CLF_DIR / "modelo_clasificacion_rf.pkl"
)

scaler_clf = joblib.load(
    ARTIFACT_CLF_DIR / "scaler_clasificacion.pkl"
)

with open(ARTIFACT_CLF_DIR / "metadata_clasificacion.json", "r", encoding="utf-8") as f:
    metadata_clf = json.load(f)

FEATURES_CLF = metadata_clf["features"]
CLASS_NAMES_CLF = metadata_clf["class_names"]

# Dropdowns de regresion



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

# prediccion de regresión

def _encode_input(input_dict):

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

    X = _encode_input(input_dict)

    X_sc = scaler_X.transform(X).astype("float32")

    pred_sc = modelo_nn.predict(X_sc, verbose=0).ravel()[0]

    pred = scaler_y.inverse_transform(

        np.array([[pred_sc]], dtype="float32")

    ).ravel()[0]

    return float(pred)

def predecir_hgb(input_dict):

    X = _encode_input(input_dict)

    X_sc = scaler_X.transform(X).astype("float32")

    pred = modelo_hgb.predict(X_sc)[0]

    return float(pred)

def predecir_clasificacion(input_dict):
    row = []

    for feat in FEATURES_CLF:
        value = input_dict.get(feat)

        if value is None:
            raise ValueError(f"Falta el valor para la variable: {feat}")

        row.append(float(value))

    X_new = np.array([row], dtype="float32")

    # Red neuronal
    X_new_s = scaler_clf.transform(X_new).astype("float32")
    proba_nn = modelo_clf_nn.predict(X_new_s, verbose=0)[0]
    pred_nn = int(np.argmax(proba_nn))

    # Random Forest
    proba_rf = modelo_clf_rf.predict_proba(X_new)[0]
    pred_rf = int(np.argmax(proba_rf))

    return {
        "pred_nn": pred_nn,
        "proba_nn": proba_nn,
        "pred_rf": pred_rf,
        "proba_rf": proba_rf,
        "label_nn": CLASS_NAMES_CLF[pred_nn],
        "label_rf": CLASS_NAMES_CLF[pred_rf],
    }

def calcular_puntaje_total_clasificacion(input_dict):
    import unicodedata
    import re

    def normalizar(texto):
        texto = str(texto).lower()
        texto = unicodedata.normalize("NFKD", texto)
        texto = "".join(c for c in texto if not unicodedata.combining(c))
        texto = re.sub(r"[^a-z0-9]", "", texto)
        return texto

    keys_norm = {
        normalizar(k): k
        for k in input_dict.keys()
    }




    total = 0

    for nombre_area, posibles_nombres in areas.items():
        key_real = None

        for nombre_norm in posibles_nombres:
            if nombre_norm in keys_norm:
                key_real = keys_norm[nombre_norm]
                break

        if key_real is None:
            raise ValueError(
                f"No se encontró la variable para calcular {nombre_area}. "
                f"Variables disponibles: {list(input_dict.keys())}"
            )

        value = input_dict.get(key_real)

        if value is None:
            raise ValueError(f"Falta el valor para calcular {nombre_area}")

        total += float(value)

    return total






# comparación modelos de regresión


def fig_comparacion_modelos():
    #MAE, MSE y R cuadrado de los experimentos
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
    # Barras de R cuadrado para cada modelo
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





# APP dash

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Saber 11 — Modelos Predictivos",
)
server = app.server 




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


# titulo
header = html.Div([
    html.H2("Modelos predictivos para el Saber 11 en el Valle del Cauca", style={"margin": 0, "fontWeight": "700"}),
    html.P("La Tupla - Jerónimo Rueda y JuanCamilo Gómez",
           style={"margin": 0, "opacity": 0.7, "fontSize": "0.9rem"}),
], style=HEADER_STYLE)

# Tabs
tabs = dcc.Tabs(
    id="tabs-main",
    value="tab-regresion",
    children=[
        dcc.Tab(
            label="Predicción de Puntaje ",
            value="tab-regresion",
            style=TAB_STYLE,
            selected_style=TAB_SELECTED_STYLE,
        ),
        dcc.Tab(
            label="Clasificación Educación Madre ",
            value="tab-clasificacion",
            style=TAB_STYLE,
            selected_style=TAB_SELECTED_STYLE,
        ),
    ],
    style={"marginBottom": "0"},
)


# Tab de regresión


def _make_dropdown(feature_name):
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
        html.H5("Predicción del puntaje", className="mb-0", style={"fontWeight": "700"}),
    ),
    dbc.CardBody(id="resultado-prediccion", children=[
        html.Div([
            html.P(
                "Ponga el perfil del estudiante que desee y espichar en 'Predecir puntjae'.",
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


    dbc.Row([
        dbc.Col(panel_inputs, lg=4, md=5, className="mb-3"),
        dbc.Col([
            panel_resultado,
            panel_modelos,
        ], lg=8, md=7),
    ]),
], className="p-3")


# Tab de clasificacion 

def crear_input_clasificacion(feature):
    label = feature.replace("_", " ").replace("punt", "Puntaje").title()

    return dbc.Col([
        dbc.Label(label, style={"fontWeight": "600", "fontSize": "0.85rem"}),
        dbc.Input(
            id=f"clf-{feature}",
            type="number",
            value=0,
            step=1,
            style={"borderRadius": "10px"}
        )
    ], md=6, className="mb-3")


tab_clasificacion = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(
                        "Clasificación del nivel educativo de la madre",
                        style={"fontWeight": "700", "color": "#1f2a3d"}
                    ),
                    html.P(
                        "Ingrese los valores del estudiante para estimar la clase educativa asociada a la madre.",
                        style={"color": "#6c757d"}
                    ),

                    dbc.Row([
                        crear_input_clasificacion(feature)
                        for feature in FEATURES_CLF
                    ]),

                    dbc.Button(
                        "Predecir clasificación",
                        id="btn-clasificar",
                        color="primary",
                        className="mt-2",
                        style={"borderRadius": "10px"}
                    )
                ])
            ], className="shadow-sm", style={"borderRadius": "18px"})
        ], lg=5),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(
                        "Resultado de clasificación",
                        style={"fontWeight": "700", "color": "#1f2a3d"}
                    ),
                    html.Div(
                        id="resultado-clasificacion",
                        children=dbc.Alert(
                            "Ingrese los valores y presione “Predecir clasificación”.",
                            color="light"
                        )
                    ),
                    html.Hr(),

                    html.H5(
                        "Evaluación de modelos",
                        style={"fontWeight": "700", "color": "#1f2a3d", "marginTop": "1rem"}
                    ),

                    html.P(
                        "Comparación entre red neuronal y Random Forest usando métricas y matrices de confusión.",
                        style={"color": "#6c757d", "fontSize": "0.9rem"}
                    ),

                    dbc.Row([
                        dbc.Col([
                            html.Img(
                                src="/assets/radar_clasificacion.png",
                                style={
                                    "width": "100%",
                                    "borderRadius": "12px",
                                    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"
                                }
                            )
                        ], lg=12, className="mb-4"),

                        dbc.Col([
                            html.Img(
                                src="/assets/matrices_confusion_clasificacion.png",
                                style={
                                    "width": "100%",
                                    "borderRadius": "12px",
                                    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"
                                }
                            )
                        ], lg=12),
                    ])
                ])
            ], className="shadow-sm", style={"borderRadius": "18px"})
        ], lg=7),
    ])
], fluid=True)


# LAYOUT

app.layout = html.Div([
    header,
    dbc.Container([
        tabs,
        html.Div(id="tab-content"),
    ], fluid=True),
])


# CALLBACKS

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

@callback(
    Output("resultado-clasificacion", "children"),
    Input("btn-clasificar", "n_clicks"),
    [State(f"clf-{feature}", "value") for feature in FEATURES_CLF],
    prevent_initial_call=True,
)
def hacer_clasificacion(n_clicks, *values):
    try:
        input_dict = dict(zip(FEATURES_CLF, values))

        resultado = predecir_clasificacion(input_dict)
        puntaje_total = calcular_puntaje_total_clasificacion(input_dict)

        proba_nn = resultado["proba_nn"]
        proba_rf = resultado["proba_rf"]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=CLASS_NAMES_CLF,
            y=proba_nn,
            name="Red neuronal",
            marker_color="#8DB5F2"
        ))

        fig.add_trace(go.Bar(
            x=CLASS_NAMES_CLF,
            y=proba_rf,
            name="Random Forest",
            marker_color="#ff7f3f"
        ))

        fig.update_layout(
            title="Probabilidad por clase",
            yaxis_title="Probabilidad",
            xaxis_title="Clase",
            yaxis=dict(range=[0, 1]),
            barmode="group",
            template="plotly_white",
            height=330,
            margin=dict(t=50, b=50, l=40, r=20),
            legend=dict(orientation="h", y=-0.25)
        )

        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H5("Resultado principal", className="alert-heading"),

                        html.Div([
                            html.Div([
                                html.H3(
                                    f"{puntaje_total:.0f} pts",
                                    style={
                                        "fontWeight": "800",
                                        "color": "#1f2a3d",
                                        "marginBottom": "0.1rem"
                                    }
                                ),
                                html.P(
                                    "Puntaje total digitado por el usuario",
                                    style={
                                        "color": "#6c757d",
                                        "fontSize": "0.9rem",
                                        "marginBottom": "0"
                                    }
                                ),
                            ], style={
                                "backgroundColor": "#f8f9fa",
                                "borderRadius": "12px",
                                "padding": "14px",
                                "marginBottom": "14px",
                                "border": "1px solid #e5e7eb"
                            }),

                            html.P([
                                "Red neuronal: ",
                                html.Strong(resultado["label_nn"])
                            ]),

                            html.P([
                                "Random Forest: ",
                                html.Strong(resultado["label_rf"])
                            ], style={"marginBottom": "0"})
                        ])
                    ], color="info")
                ], lg=12)
            ]),

            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False}
            )
        ])

    except Exception as e:
        return dbc.Alert(
            [
                html.Strong("Error al generar la clasificación: "),
                html.Code(str(e))
            ],
            color="danger"
        )

# MAIN

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
 