import streamlit as st
from utils.model_utils import load_model

MODEL_PATH = "app/model/modelo_lego_final.pth"
model = load_model(MODEL_PATH)

st.set_page_config(page_title="Identificación de Sets LEGO", layout="wide")

st.title("🧩 Identificación de Sets de LEGO")
st.sidebar.image("app/static/logo.png", width=150)

st.markdown(
    """
    🎯 **Esta aplicación permite identificar sets de LEGO a partir de imágenes.**
    - 📸 **Sube una imagen del set de LEGO**
    - 📹 **Usa la cámara en tiempo real**
    - 📊 **Consulta el historial de predicciones**
    """
)

st.subheader("🔍 Selecciona una opción en el menú de la izquierda.")
