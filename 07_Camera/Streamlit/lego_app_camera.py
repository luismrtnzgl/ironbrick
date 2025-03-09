import streamlit as st
import torch
import os
import urllib.request
import pandas as pd
import json
import asyncio
from model_utils import load_model
from predict import predict

# 🔥 Solución para evitar el error "no running event loop" en Streamlit.io
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="Identificación de Sets LEGO", layout="wide")

# 📌 Rutas de archivos locales
MODEL_PATH = "modelo_lego_final.pth"
MAPPING_PATH = "idx_to_class.json"
DATA_PATH = "scraped_lego_data.csv"

# 📌 URLs de los archivos en GitHub (deben estar en formato RAW)
MODEL_URL = "https://github.com/luismrtnzgl/ironbrick/raw/refs/heads/main/07_Camera/Streamlit/modelo_lego_final.pth"
MAPPING_URL = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/refs/heads/main/07_Camera/Streamlit/idx_to_class.json"
DATASET_URL = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/refs/heads/main/04_Extra/APP/data/scraped_lego_data.csv"

# 🔥 Descargar archivos si no existen
def download_file(url, path):
    if not os.path.exists(path):
        st.write(f"📥 Descargando {path} desde GitHub...")
        urllib.request.urlretrieve(url, path)
        st.write(f"✅ {path} descargado exitosamente.")

download_file(MODEL_URL, MODEL_PATH)
download_file(MAPPING_URL, MAPPING_PATH)
download_file(DATASET_URL, DATA_PATH)

# 📌 Cargar el modelo
model = load_model(MODEL_PATH)

# 📌 Cargar el mapeo de clases a sets
if os.path.exists(MAPPING_PATH):
    with open(MAPPING_PATH, "r") as f:
        idx_to_class = json.load(f)
else:
    st.error("❌ Error: No se encontró el archivo de mapeo idx_to_class.json.")
    idx_to_class = {}

# 📌 Cargar datos de LEGO
if os.path.exists(DATA_PATH):
    df_lego = pd.read_csv(DATA_PATH)
    df_lego["Number"] = df_lego["Number"].astype(str)  # ✅ Asegura que los números de set sean strings
else:
    st.error("❌ Error: El archivo scraped_lego_data.csv no se encontró.")
    df_lego = None

st.title("🧩 Identificación de Sets de LEGO")

st.markdown(
    """
    🎯 **Sube una imagen o usa la cámara para identificar un set de LEGO.**
    - 📸 **Sube una imagen**
    - 📹 **Usa la cámara**
    """
)

# 📷 Captura de imagen
uploaded_file = st.file_uploader("Sube una imagen del set de LEGO", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    from PIL import Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_container_width=True)  # ✅ Corrección de `use_column_width`

    # 🔥 Realizar predicción
    predicted_class, probabilities = predict(image, model)
    confidence = probabilities[0][predicted_class] * 100  # Convertir a porcentaje

    # Convertir la predicción al número de set real
    predicted_set_number = idx_to_class.get(str(predicted_class), "Desconocido")

    # Buscar información en el dataset si está disponible
    if df_lego is not None:
        set_info = df_lego[df_lego["Number"] == predicted_set_number]

        if not set_info.empty:
            set_name = set_info.iloc[0].get('SetName', 'Desconocido')
            used_price = set_info.iloc[0].get('CurrentValueUsed', 'N/A')

            st.subheader(f"🔍 Set identificado: {set_name} ({predicted_set_number})")
            st.write(f"📈 Confianza: {confidence:.2f}%")
            st.write(f"💰 Precio usado estimado: ${used_price}")
        else:
            st.warning(f"⚠️ No se encontró información del set {predicted_set_number}.")
    else:
        st.error("❌ No se puede mostrar información del set porque el dataset no está disponible.")
