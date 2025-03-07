import streamlit as st
import torch
import os
import urllib.request
from model_utils import load_model
from predict import predict

# 📌 Ruta del modelo en la máquina de Streamlit
MODEL_PATH = "modelo_lego_final.pth"

# 📌 URL del modelo en GitHub (Asegúrate de usar el enlace RAW del archivo)
MODEL_URL = "https://github.com/luismrtnzgl/ironbrick/raw/93ee1070fbea6c5b42724b2e0bb4e9236bff2966/07_Camera/Streamlit/modelo_lego_final.pth"

# 🔥 Descargar el modelo si no existe
if not os.path.exists(MODEL_PATH):
    st.write("📥 Descargando modelo desde GitHub...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.write("✅ Modelo descargado exitosamente.")

# 📌 Cargar el modelo
model = load_model(MODEL_PATH)

st.set_page_config(page_title="Identificación de Sets LEGO", layout="wide")

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

    st.subheader(f"🔍 Set identificado: {predicted_class}")
    st.write(f"📈 Confianza: {confidence:.2f}%")
