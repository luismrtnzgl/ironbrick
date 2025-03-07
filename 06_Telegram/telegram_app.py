import streamlit as st
import psycopg2
import pandas as pd
import joblib
import requests
import os
import numpy as np

# 📌 Obtener la URL de la base de datos PostgreSQL desde Render
DB_URL = os.getenv("DATABASE_URL")

# 📌 Función para conectar con la base de datos en Render
def get_db_connection():
    return psycopg2.connect(DB_URL, sslmode="require")

# 📌 Cargar el modelo
modelo_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/05_Streamlit/models/stacking_model.pkl"

@st.cache_resource
def cargar_modelo():
    modelo_path = "/tmp/stacking_model.pkl"
    
    if not os.path.exists(modelo_path):
        response = requests.get(modelo_url)
        with open(modelo_path, "wb") as f:
            f.write(response.content)
    
    return joblib.load(modelo_path)

modelo = cargar_modelo()

# 📌 Cargar y procesar el dataset de LEGO
dataset_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/01_Data_Cleaning/df_lego_final_venta.csv"

@st.cache_data
def cargar_datos():
    df = pd.read_csv(dataset_url)
    return preprocess_data(df)

def preprocess_data(df):
    df = df[df['USRetailPrice'] > 0].copy()

    # 📌 Asegurar que estas columnas existen antes de mapear
    if 'Exclusivity' in df.columns:
        exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
        df['Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)

    if 'SizeCategory' in df.columns:
        size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
        df['SizeCategory'] = df['SizeCategory'].map(size_category_mapping)

    # 📌 Crear métricas solo si las columnas existen
    if 'Pieces' in df.columns and 'USRetailPrice' in df.columns:
        df["PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]

    if 'Minifigs' in df.columns and 'USRetailPrice' in df.columns:
        df["PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)

    if 'ExitYear' in df.columns and 'LaunchYear' in df.columns:
        df["YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]

    # 📌 Filtrar solo columnas numéricas antes de limpiar datos
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # 📌 Reemplazar valores infinitos por NaN y luego llenarlos con la mediana
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

df_lego = cargar_datos()

# 📌 Formulario para guardar configuración del usuario
st.title("📢 Alerta mensual de Inversión en LEGO por Telegram")
st.write("Registra tus preferencias para recibir propuestas de inversión por Telegram cada mes.")

telegram_id = st.text_input("🔹 Tu ID de Telegram (usa @userinfobot en Telegram para obtenerlo)")
presupuesto_min, presupuesto_max = st.slider("💰 Rango de presupuesto (USD)", 10, 500, (10, 200), step=10)

temas_unicos = sorted(df_lego["Theme"].unique().tolist())
temas_opciones = ["Todos"] + temas_unicos
temas_favoritos = st.multiselect("🛒 Temas Favoritos", temas_opciones, default=["Todos"])

if st.button("💾 Guardar configuración"):
    temas_str = ",".join(temas_favoritos)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        telegram_id TEXT PRIMARY KEY,
        presupuesto_min INTEGER,
        presupuesto_max INTEGER,
        temas_favoritos TEXT
    )
    """)

    cursor.execute("""
    INSERT INTO usuarios (telegram_id, presupuesto_min, presupuesto_max, temas_favoritos)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (telegram_id) DO UPDATE
    SET presupuesto_min = EXCLUDED.presupuesto_min,
        presupuesto_max = EXCLUDED.presupuesto_max,
        temas_favoritos = EXCLUDED.temas_favoritos;
    """, (telegram_id, presupuesto_min, presupuesto_max, temas_str))
    
    conn.commit()
    conn.close()
    st.success("✅ ¡Tus preferencias han sido guardadas correctamente!")

# 📌 Mostrar los mejores sets de inversión según el modelo
st.write("📊 **Top Sets Recomendados por el Modelo**:")
# 📌 Aplicar el modelo de predicción antes de mostrar el ranking
features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 
            'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity', 
            'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

df_lego["PredictedInvestmentScore"] = modelo.predict(df_lego[features])

# 📌 Mostrar los mejores sets de inversión según el modelo
st.write("📊 **Top 10 Sets Recomendados por el Modelo**:")
df_recomendados = df_lego.sort_values(by="PredictedInvestmentScore", ascending=False).head(10)
st.dataframe(df_recomendados[["SetName", "Number", "USRetailPrice", "PredictedInvestmentScore", "Theme"]])


# 📌 Mostrar usuarios registrados
st.write("📊 **Usuarios Registrados en la Base de Datos**")

conn = get_db_connection()
cursor = conn.cursor()

cursor.execute("SELECT telegram_id, presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios")
usuarios = cursor.fetchall()

if usuarios:
    df_usuarios = pd.DataFrame(usuarios, columns=["Telegram ID", "Presupuesto Mín", "Presupuesto Máx", "Temas Favoritos"])
    st.dataframe(df_usuarios)
else:
    st.warning("❌ No hay usuarios registrados en el bot.")

conn.close()
