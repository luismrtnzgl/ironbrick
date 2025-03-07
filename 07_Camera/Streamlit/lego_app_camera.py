import streamlit as st
import torch
import os
import urllib.request
import pandas as pd
from model_utils import load_model
from predict import predict

st.set_page_config(page_title="Identificación de Sets LEGO", layout="wide")

# 📌 Ruta del modelo en la máquina de Streamlit
MODEL_PATH = "modelo_lego_final.pth"
state_dict = torch.load(MODEL_PATH, map_location="cpu")

# 📌 URL del modelo en GitHub
MODEL_URL = "https://github.com/luismrtnzgl/ironbrick/raw/93ee1070fbea6c5b42724b2e0bb4e9236bff2966/07_Camera/Streamlit/modelo_lego_final.pth"

# 🔥 Descargar el modelo si no existe
if not os.path.exists(MODEL_PATH):
    st.write("📥 Descargando modelo desde GitHub...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.write("✅ Modelo descargado exitosamente.")

# 📌 Cargar el modelo
model = load_model(MODEL_PATH)

# 📌 Cargar datos de LEGO
DATA_PATH = "scraped_lego_data.csv"
if os.path.exists(DATA_PATH):
    df_lego = pd.read_csv(DATA_PATH)
else:
    st.error("❌ Error: El archivo scraped_lego_data.csv no se encontró.")
    df_lego = None  # Para evitar errores si el archivo falta

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
    st.image(image, caption="Imagen subida", use_column_width=True)

    # 🔥 Realizar predicción
    predicted_class, confidence = predict(image, model)

    # Buscar información en el dataset si está disponible
    if df_lego is not None:
        set_info = df_lego[df_lego.index == predicted_class]
        if not set_info.empty:
            set_number = set_info.iloc[0]['Number']
            set_name = set_info.iloc[0]['SetName']
            used_price = set_info.iloc[0]['CurrentValueUsed']
        else:
            set_number = "Desconocido"
            set_name = "Desconocido"
            used_price = "N/A"

        st.subheader(f"🔍 Set identificado: {set_name} ({set_number})")
        st.write(f"📈 Confianza: {confidence:.2f}%")
        st.write(f"💰 Precio usado estimado: ${used_price}")
    else:
        st.error("❌ No se puede mostrar información del set porque el dataset no está disponible.")
