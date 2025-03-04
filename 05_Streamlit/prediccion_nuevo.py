import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# 🔹 Función de preprocesamiento (la misma que usaste en el entrenamiento)
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

    # 🔹 Aplicar preprocesamiento para asegurar que las columnas necesarias existen
    df = preprocess_data(df)

    # 🔹 Verificar si el modelo genera "PredictedInvestmentScore"
    if "PredictedInvestmentScore" not in df.columns:
        st.warning("⚠️ No se encontró 'PredictedInvestmentScore', aplicando modelo...")
        model = load_model()
        features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 'ResaleDemand', 
                    'AnnualPriceIncrease', 'Exclusivity', 'SizeCategory', 'PricePerPiece', 
                    'PricePerMinifig', 'YearsOnMarket', 'InteractionFeature']
        
        try:
            df["PredictedInvestmentScore"] = model.predict(df[features])
            st.success("✅ PredictedInvestmentScore generado correctamente.")
        except Exception as e:
            st.error(f"❌ Error al generar PredictedInvestmentScore: {e}")
            st.stop()

    return df

# 🔹 Cargar recursos
df_ranking = load_data()

# 🔹 Streamlit App
st.title("Plataforma de Recomendación de Inversión en LEGO 📊")

# 🔹 Entrada del usuario
st.subheader("Configura tu Inversión en LEGO")
presupuesto = st.number_input("Presupuesto (USD)", min_value=100.0, max_value=3000.0, value=500.0)
themes = st.multiselect("Selecciona los Themes de Interés", options=df_ranking["Theme"].unique(), default=df_ranking["Theme"].unique())

# 🔹 Filtrar sets basados en themes seleccionados
df_filtrado = df_ranking[df_ranking["Theme"].isin(themes)].copy()

# 🔹 Verificar si "PredictedInvestmentScore" está en df_filtrado
if "PredictedInvestmentScore" not in df_filtrado.columns:
    st.error("❌ Error: La columna 'PredictedInvestmentScore' no está en el DataFrame filtrado.")
    st.write("Columnas disponibles:", df_filtrado.columns.tolist())
    st.stop()

# 🔹 Generar combinaciones de inversión
st.subheader("Opciones de Inversión")

def generar_opciones_inversion(df, presupuesto, n_opciones=3):
    opciones = []
    for _ in range(n_opciones):
        df_sample = df.sample(frac=1).reset_index(drop=True)  # Mezclar sets aleatoriamente
        inversion = []
        total = 0
        for _, row in df_sample.iterrows():
            if total + row["USRetailPrice"] <= presupuesto:
                inversion.append(row)
                total += row["USRetailPrice"]

        df_opcion = pd.DataFrame(inversion)
        
        # Verificar columnas
        if "PredictedInvestmentScore" not in df_opcion.columns:
            st.error("❌ Error: 'PredictedInvestmentScore' se eliminó en la selección.")
            st.write("Columnas en df_opcion:", df_opcion.columns.tolist())
            continue  # Saltar esta opción

        opciones.append(df_opcion)

    return opciones

# 🔹 Generar 3 opciones de inversión
opciones = generar_opciones_inversion(df_filtrado, presupuesto, n_opciones=3)
colores = ["#FF5733", "#FFC300", "#28B463"]  # Rojo, Amarillo, Verde según PredictedInvestmentScore

for i, df_opcion in enumerate(opciones):
    if not df_opcion.empty:
        score_promedio = df_opcion["PredictedInvestmentScore"].mean()
        color = colores[0] if score_promedio < 8 else colores[1] if score_promedio < 15 else colores[2]
        st.markdown(f"""
            <div style='background-color:{color}; padding:10px; border-radius:5px; text-align:center;'>
                <strong>Score Promedio: {score_promedio:.2f}</strong>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f"### Opción {i+1} - Inversión Total: ${df_opcion['USRetailPrice'].sum():.2f}")
        st.dataframe(df_opcion[["SetName", "Theme", "USRetailPrice", "PredictedInvestmentScore"]])

# 🔹 Visualización de los mejores sets
st.subheader("Top 10 Sets con Mejor Potencial de Inversión")
df_top = df_ranking.sort_values(by="PredictedInvestmentScore", ascending=False).head(10)
fig = px.bar(df_top, x="SetName", y="PredictedInvestmentScore", color="PredictedInvestmentScore", color_continuous_scale=["red", "yellow", "green"])
st.plotly_chart(fig)
