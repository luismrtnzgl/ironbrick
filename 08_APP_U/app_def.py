import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import pymongo
import psycopg2

# Configuración de la app
st.set_page_config(page_title="IronbrickML", page_icon="🧱", layout="wide")

# Sidebar para la navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Selecciona una página", ["Recomendador de Inversión", "Alertas de Telegram"])

# Conexión a MongoDB
@st.cache_resource
def init_mongo_connection():
    return pymongo.MongoClient(st.secrets["mongo"]["uri"])

mongo_client = init_mongo_connection()
mongo_db = mongo_client[st.secrets["mongo"]["db"]]
mongo_collection = mongo_db[st.secrets["mongo"]["collection"]]

# Conexión a PostgreSQL
#DB_URL = os.getenv("DATABASE_URL")

#def get_db_connection():
    #return psycopg2.connect(DB_URL, sslmode="require")

# Cargar modelo de predicción
modelo_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/05_Streamlit/models/stacking_model.pkl"

@st.cache_resource
def load_model():
    modelo_path = "/tmp/stacking_model.pkl"
    if not os.path.exists(modelo_path):
        response = requests.get(modelo_url)
        with open(modelo_path, "wb") as f:
            f.write(response.content)
    return joblib.load(modelo_path)

modelo = load_model()

# Cargar datos desde MongoDB
@st.cache_data(ttl=600)
def load_data():
    data = list(mongo_collection.find({}, {"_id": 0}))
    if not data:
        st.error("❌ No se encontraron datos en MongoDB.")
        st.stop()
    df = pd.DataFrame(data)
    return preprocess_data(df)

def preprocess_data(df):
    df = df[df['USRetailPrice'] > 0].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    return df

df_lego = load_data()

if page == "Recomendador de Inversión":
    st.title("Recomendador de Inversión en LEGO 📊")
    presupuesto_min, presupuesto_max = st.slider("💰 Rango de presupuesto (USD)", 10, 1000, (10, 200), step=10)
    temas_unicos = sorted(df_lego["Theme"].unique().tolist())
    temas_opciones = ["Todos"] + temas_unicos
    selected_themes = st.multiselect("🛒 Selecciona los Themes de Interés", temas_opciones, default=["Todos"])

    df_filtrado = df_lego[(df_lego["USRetailPrice"] >= presupuesto_min) & (df_lego["USRetailPrice"] <= presupuesto_max)]
    if "Todos" not in selected_themes:
        df_filtrado = df_filtrado[df_filtrado["Theme"].isin(selected_themes)]

    if df_filtrado.empty:
        st.error("❌ No hay sets disponibles con los filtros seleccionados.")
        st.stop()

    if st.button("Generar Predicciones"):
        features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit',
                    'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity',
                    'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']
        df_filtrado["PredictedInvestmentScore"] = modelo.predict(df_filtrado[features])
        df_filtrado = df_filtrado[df_filtrado["PredictedInvestmentScore"] > 0].sort_values(by="PredictedInvestmentScore", ascending=False).head(3)
        st.subheader("📊 Top 3 Sets Más Rentables")
        st.write(df_filtrado[["SetName", "Theme", "USRetailPrice", "PredictedInvestmentScore"]])

elif page == "Alertas de Telegram":
    st.title("📢 Configuración de Alertas de Telegram")
    telegram_id = st.text_input("🔹 Tu ID de Telegram (@userinfobot)")
    presupuesto_min, presupuesto_max = st.slider("💰 Rango de presupuesto (USD)", 10, 500, (10, 200), step=10)
    temas_favoritos = st.multiselect("🛒 Temas Favoritos", temas_opciones, default=["Todos"])

    if st.button("💾 Alta en Alertas"):
        temas_str = ",".join(temas_favoritos)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            telegram_id TEXT PRIMARY KEY,
            presupuesto_min INTEGER,
            presupuesto_max INTEGER,
            temas_favoritos TEXT
        )""")
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
        st.success("✅ Preferencias guardadas correctamente!")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT telegram_id, presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios")
    usuarios = cursor.fetchall()
    conn.close()

    if usuarios:
        df_usuarios = pd.DataFrame(usuarios, columns=["Telegram ID", "Presupuesto Mín", "Presupuesto Máx", "Temas Favoritos"])
        st.dataframe(df_usuarios)
    else:
        st.warning("❌ No hay usuarios registrados.")
