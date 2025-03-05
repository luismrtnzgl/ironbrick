import streamlit as st
import sqlite3
import pandas as pd
import joblib
import requests
import os

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
    return pd.read_csv(dataset_url)

# 📌 Cargar dataset ANTES de hacer predicciones
df_lego = cargar_datos()

# 📌 Verificar que df_lego está cargado antes de usarlo
if df_lego is not None and not df_lego.empty:
    # 📌 Hacer predicción con el modelo
    features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 
                'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity', 
                'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

    if all(feature in df_lego.columns for feature in features):
        df_lego["PredictedInvestmentScore"] = modelo.predict(df_lego[features])

        # 📌 Mostrar los mejores sets de inversión según el modelo
        st.write("📊 **Top Sets Recomendados por el Modelo**:")
        df_recomendados = df_lego.sort_values(by="PredictedInvestmentScore", ascending=False).head(10)
        st.dataframe(df_recomendados)
    else:
        st.error("❌ Algunas columnas faltan en el dataset. Verifica que todas las características estén disponibles.")
else:
    st.error("❌ No se pudo cargar el dataset. Verifica la URL de GitHub.")

# 📌 Base de datos SQLite
conn = sqlite3.connect("user_ironbrick.db", check_same_thread=False)
cursor = conn.cursor()

# 📌 Crear tabla para guardar configuraciones de inversión
cursor.execute("""
CREATE TABLE IF NOT EXISTS usuarios (
    telegram_id TEXT PRIMARY KEY,
    presupuesto_max INTEGER,
    temas_favoritos TEXT,
    rentabilidad_min INTEGER,
    piezas_min INTEGER,
    exclusivo TEXT
)
""")
conn.commit()

# 📌 Interfaz en Streamlit
st.title("📢 Configuración de Alertas de Inversión en LEGO")

st.write("Registra tus preferencias para recibir propuestas de inversión en Telegram cada mes.")

telegram_id = st.text_input("🔹 Tu ID de Telegram (usa @userinfobot en Telegram para obtenerlo)")
presupuesto_max = st.number_input("💰 Presupuesto Máximo (USD)", min_value=10, value=200)
temas_favoritos = st.multiselect("🛒 Temas Favoritos", ["Star Wars", "Technic", "Creator Expert", "Harry Potter"])
rentabilidad_min = st.slider("📈 Rentabilidad esperada en 2 años (%)", 10, 100, 30)
piezas_min = st.number_input("🔢 Cantidad mínima de piezas", min_value=50, value=500)
exclusivo = st.checkbox("🔒 Solo sets exclusivos")

# 📌 Guardar información en la base de datos
if st.button("💾 Guardar configuración"):
    temas_str = ",".join(temas_favoritos)
    exclusivo_str = "Sí" if exclusivo else "No"
    
    cursor.execute("""
    INSERT OR REPLACE INTO usuarios (telegram_id, presupuesto_max, temas_favoritos, rentabilidad_min, piezas_min, exclusivo)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (telegram_id, presupuesto_max, temas_str, rentabilidad_min, piezas_min, exclusivo_str))
    
    conn.commit()
    st.success("✅ ¡Tus preferencias han sido guardadas! Recibirás alertas mensuales en Telegram.")

# 📌 Hacer predicción con el modelo
df_lego["PredictedInvestmentScore"] = modelo.predict(df_lego[['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 
                                                               'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity', 
                                                               'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']])

# 📌 Mostrar los mejores sets de inversión según el modelo
st.write("📊 **Top Sets Recomendados por el Modelo**:")
df_recomendados = df_lego.sort_values(by="PredictedInvestmentScore", ascending=False).head(10)
st.dataframe(df_recomendados)
