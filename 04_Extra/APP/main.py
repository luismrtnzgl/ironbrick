import os
import streamlit as st
import pandas as pd

# 📌 Obtener la ruta del archivo CSV
BASE_DIR = os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "04_Extra/APP/data/scraped_lego_data.csv")

# 📌 Verificar si el archivo existe
st.write("📂 Ruta del archivo CSV:", CSV_PATH)
if not os.path.exists(CSV_PATH):
    st.error("❌ ERROR: El archivo CSV NO EXISTE en la ruta especificada.")
    st.stop()

# 📌 Cargar el archivo CSV sin modificaciones
try:
    df = pd.read_csv(CSV_PATH)
    st.success("✅ Archivo CSV cargado correctamente.")
    st.write("📏 Dimensiones del archivo:", df.shape)
    st.dataframe(df.head(20))  # Mostrar las primeras 20 filas
except Exception as e:
    st.error(f"❌ ERROR al leer el archivo CSV: {e}")
    st.stop()
