import streamlit as st
import psycopg2
import pandas as pd
import joblib
import requests
import os
import numpy as np
import pymongo

# Obtenemos la URL de la base de datos PostgreSQL desde Render
DB_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DB_URL, sslmode="require")

@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo"]["uri"])

client = init_connection()
db = client[st.secrets["mongo"]["db"]]
collection = db[st.secrets["mongo"]["collection"]]

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

@st.cache_data(ttl=600)
def load_data():
    data = list(collection.find({}, {"_id": 0}))
    if not data:
        st.error("❌ No se encontraron datos en la colección de MongoDB.")
        st.stop()
    df = pd.DataFrame(data)
    return preprocess_data(df)

def preprocess_data(df):
    df = df[df['USRetailPrice'] > 0].copy()

    if 'Exclusivity' in df.columns:
        exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
        df['Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)

    if 'SizeCategory' in df.columns:
        size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
        df['SizeCategory'] = df['SizeCategory'].map(size_category_mapping)

    # Asegurar la creación de columnas faltantes
    df["PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]
    df["PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)
    df["YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]

    # Llenar valores NaN o Inf con 0
    df.replace([np.inf, -np.inf], 0, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

df_lego = load_data()

st.title("📢 Alerta mensual de Inversión en LEGO por Telegram")
st.write("**Bienvenido a IronbrickML - Alertas de Inversión en LEGO**")
st.write(
    "📊 IronbrickML analiza la rentabilidad de sets de LEGO utilizando modelos de predicción de inversión. "
    "Cada mes, recibirás una recomendación personalizada en Telegram con el set que mejor se ajuste a tu presupuesto y preferencias. "
    "Solo se te sugerirán sets con alto potencial de revalorización y sin repeticiones para que siempre tengas nuevas oportunidades de inversión. "
    "Configura tus preferencias y deja que la inteligencia artificial haga el resto."
)

telegram_id = st.text_input("🔹 Tu ID de Telegram (usa @userinfobot en Telegram para obtenerlo)")
presupuesto_min, presupuesto_max = st.slider("💰 Rango de presupuesto (USD)", 10, 500, (10, 200), step=10)

temas_unicos = sorted(df_lego["Theme"].unique().tolist())
temas_opciones = ["Todos"] + temas_unicos
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

features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit',
            'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity',
            'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

df_lego["PredictedInvestmentScore"] = modelo.predict(df_lego[features])

# Transformamos los valores de revalorización en categorías
def clasificar_revalorizacion(score):
    if score > 13:
        return "Muy Alta"
    elif 10 <= score <= 13:
        return "Alta"
    elif 5 <= score < 10:
        return "Media"
    elif 0 <= score < 5:
        return "Baja"
    else:
        return "Ninguna"

df_lego["Revalorización"] = df_lego["PredictedInvestmentScore"].apply(clasificar_revalorizacion)

df_lego.rename(columns={
    "Number": "Set",
    "SetName": "Nombre",
    "USRetailPrice": "Precio",
    "Theme": "Tema"
}, inplace=True)

st.write("📊 **Sets Recomendados por IronbrickML**:")
df_recomendados = df_lego[df_lego["PredictedInvestmentScore"] > 0].sort_values(by="PredictedInvestmentScore", ascending=False)
st.data_editor(df_recomendados[["Set", "Nombre", "Precio", "Tema", "Revalorización"]], disabled=True)

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