import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 🔹 Función de preprocesamiento (para asegurar que las columnas están presentes)
def preprocess_data(df):
    df = df[df['USRetailPrice'] > 0].copy()

    exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
    df.loc[:, 'Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)

    size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
    df.loc[:, 'SizeCategory'] = df['SizeCategory'].map(size_category_mapping)

    df.loc[:, "PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]
    df.loc[:, "PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)
    df.loc[:, "YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]
    df.loc[:, "InteractionFeature"] = df["PricePerPiece"] * df["YearsOnMarket"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

# 🔹 Definir rutas absolutas
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models/stacking_model_compressed.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data/df_lego_final_venta.csv")

# 🔹 Cargar el modelo con joblib
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# 🔹 Cargar y procesar los datos
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # 🔹 Aplicar preprocesamiento
    df = preprocess_data(df)

    # 🔹 Generar PredictedInvestmentScore si no está presente
    if "PredictedInvestmentScore" not in df.columns:
        st.warning("⚠️ No se encontró 'PredictedInvestmentScore', aplicando modelo...")
        model = load_model()
        features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 'ResaleDemand', 
                    'AnnualPriceIncrease', 'Exclusivity', 'SizeCategory', 'PricePerPiece', 
                    'PricePerMinifig', 'YearsOnMarket', 'InteractionFeature']
        
        df["PredictedInvestmentScore"] = model.predict(df[features])
        st.success("✅ PredictedInvestmentScore generado correctamente.")

    return df

# 🔹 Cargar recursos
df_ranking = load_data()

# 🔹 Streamlit App
st.title("Plataforma de Recomendación de Inversión en LEGO 📊")

# 🔹 Selección de presupuesto con rango
st.subheader("Configura tu Inversión en LEGO")
presupuesto_min, presupuesto_max = st.slider("Selecciona el rango de presupuesto (USD)", 
                                             min_value=100, max_value=3000, value=(500, 1500), step=50)

# 🔹 Selección de temas con opción "Todos"
themes_options = ["Todos"] + sorted(df_ranking["Theme"].unique().tolist())
selected_themes = st.multiselect("Selecciona los Themes de Interés", themes_options, default=["Todos"])

# 🔹 Filtrar sets según selección de temas
if "Todos" in selected_themes:
    df_filtrado = df_ranking
else:
    df_filtrado = df_ranking[df_ranking["Theme"].isin(selected_themes)].copy()

# 🔹 Filtrar por presupuesto
df_filtrado = df_filtrado[(df_filtrado["USRetailPrice"] >= presupuesto_min) & 
                          (df_filtrado["USRetailPrice"] <= presupuesto_max)]

# 🔹 Filtrar sets con un `PredictedInvestmentScore` mayor a 1
df_filtrado = df_filtrado[df_filtrado["PredictedInvestmentScore"] > 1]

# 🔹 Generar combinaciones de inversión
st.subheader("Opciones de Inversión")

def generar_opciones_inversion(df, n_opciones=3):
    opciones = []
    df_sorted = df.sort_values(by="PredictedInvestmentScore", ascending=False)  # Priorizar mejores sets

    for _ in range(n_opciones):
        inversion = []
        total = 0
        for _, row in df_sorted.iterrows():
            if len(inversion) < 3:  # Máximo 3 sets por opción
                inversion.append(row)
                total += row["USRetailPrice"]

        df_opcion = pd.DataFrame(inversion)

        if "PredictedInvestmentScore" not in df_opcion.columns:
            continue  # Si falta la columna, descartar la opción

        opciones.append(df_opcion)

    return opciones

# 🔹 Generar 3 opciones de inversión
opciones = generar_opciones_inversion(df_filtrado, n_opciones=3)

# 🔹 Definir colores para la seguridad de la inversión
def get_color(score):
    if score > 15:
        return "#28B463"  # Verde
    elif score > 6:
        return "#FFC300"  # Amarillo
    else:
        return "#FF5733"  # Naranja

for i, df_opcion in enumerate(opciones):
    if not df_opcion.empty:
        score_promedio = df_opcion["PredictedInvestmentScore"].mean()
        color = get_color(score_promedio)

        st.markdown(f"""
            <div style='background-color:{color}; padding:10px; border-radius:5px; text-align:center;'>
                <strong>Score Promedio: {score_promedio:.2f}</strong>
            </div>
        """, unsafe_allow_html=True)

        for _, row in df_opcion.iterrows():
            st.image(row["BricksetImageURL"], width=150)
            st.markdown(f"### [{row['SetName']}]({row['BricksetURL']})")
            st.write(f"💰 **Precio:** ${row['USRetailPrice']:.2f}")
            st.write(f"📊 **Predicted Investment Score:** {row['PredictedInvestmentScore']:.2f}")
            st.write("---")  # Línea separadora entre sets
