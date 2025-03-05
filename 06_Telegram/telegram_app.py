import streamlit as st
import sqlite3
import pandas as pd
import joblib
import requests
import os
import numpy as np

# 📌 URL del modelo en GitHub RAW
modelo_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/05_Streamlit/models/stacking_model.pkl"

@st.cache_resource
def cargar_modelo():
    """Descarga el modelo desde GitHub y lo carga en Streamlit Cloud."""
    modelo_path = "/tmp/stacking_model.pkl"  # Ruta temporal en Streamlit Cloud
    
    # 📌 Descargar el archivo si no existe
    if not os.path.exists(modelo_path):
        response = requests.get(modelo_url)
        with open(modelo_path, "wb") as f:
            f.write(response.content)
    
    # 📌 Cargar el modelo
    return joblib.load(modelo_path)

# 📌 Cargar el modelo ANTES de usarlo
modelo = cargar_modelo()

# 📌 URL del dataset en GitHub RAW
dataset_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/01_Data_Cleaning/df_lego_final_venta.csv"

@st.cache_data
def cargar_datos():
    df = pd.read_csv(dataset_url)
    return preprocess_data(df)  # Aplicar preprocesamiento antes de usarlo

# 📌 Función de preprocesamiento
def preprocess_data(df):
    df = df[df['USRetailPrice'] > 0].copy()

    exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
    df['Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)

    size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
    df['SizeCategory'] = df['SizeCategory'].map(size_category_mapping)

    df["PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]
    df["PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)
    df["YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

# 📌 Cargar dataset con preprocesamiento
df_lego = cargar_datos()

# 📌 Obtener temas únicos y agregar opción "Todos"
temas_unicos = sorted(df_lego["Theme"].unique().tolist())
temas_opciones = ["Todos"] + temas_unicos  # Agregar opción "Todos"

# 📌 Interfaz en Streamlit
st.title("📢 Alerta mensual de Inversión en LEGO por Telegram")

st.write("Registra tus preferencias para recibir propuestas de inversión por Telegram cada mes.")

telegram_id = st.text_input("🔹 Tu ID de Telegram (usa @userinfobot en Telegram para obtenerlo)")

# 📌 Slider de Presupuesto Máximo con rango
presupuesto_min, presupuesto_max = st.slider("💰 Rango de presupuesto (USD)", 10, 500, (10, 200), step=10)

# 📌 Selección de temas dinámicos desde el dataset con opción "Todos" preseleccionada
temas_favoritos = st.multiselect("🛒 Temas Favoritos", temas_opciones, default=["Todos"])

# 📌 Guardar información en la base de datos
if st.button("💾 Guardar configuración"):
    temas_str = ",".join(temas_favoritos)
    
    cursor = sqlite3.connect("user_ironbrick.db", check_same_thread=False).cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        telegram_id TEXT PRIMARY KEY,
        presupuesto_min INTEGER,
        presupuesto_max INTEGER,
        temas_favoritos TEXT
    )
    """)
    cursor.execute("""
    INSERT OR REPLACE INTO usuarios (telegram_id, presupuesto_min, presupuesto_max, temas_favoritos)
    VALUES (?, ?, ?, ?)
    """, (telegram_id, presupuesto_min, presupuesto_max, temas_str))
    
    cursor.connection.commit()
    st.success("✅ ¡Tus preferencias han sido guardadas! Recibirás alertas mensuales en Telegram.")

# 📌 Hacer predicción con el modelo y mostrar resultados
st.write("📊 **Top Sets Recomendados por el Modelo**:")

features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 
            'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity', 
            'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

df_lego["PredictedInvestmentScore"] = modelo.predict(df_lego[features])

# 📌 Filtrar por presupuesto y temas seleccionados
df_filtrado = df_lego[(df_lego["USRetailPrice"] >= presupuesto_min) & 
                       (df_lego["USRetailPrice"] <= presupuesto_max)]

if "Todos" not in temas_favoritos:
    df_filtrado = df_filtrado[df_filtrado["Theme"].isin(temas_favoritos)]

# 📌 Seleccionar solo las columnas deseadas y renombrarlas
df_recomendados = df_filtrado[["Number", "Theme", "SetName", "USRetailPrice", "WantCount", "PredictedInvestmentScore"]].copy()

# 📌 Renombrar las columnas
df_recomendados.rename(columns={
    "Number": "ID",
    "Theme": "Tema",
    "SetName": "Nombre del set",
    "USRetailPrice": "Precio de compra",
    "WantCount": "Personas que lo quieren",
    "PredictedInvestmentScore": "Rentabilidad como inversión"
}, inplace=True)

# 📌 Convertir la rentabilidad en categorías de texto
def clasificar_rentabilidad(score):
    if score > 10:
        return "Alta"
    elif 5 <= score <= 10:
        return "Media"
    else:
        return "Baja"

df_recomendados["Rentabilidad como inversión"] = df_recomendados["Rentabilidad como inversión"].apply(clasificar_rentabilidad)

# 📌 Mostrar la tabla con los resultados
st.dataframe(df_recomendados)
