import streamlit as st
from utils.model_utils import load_model
from app.predict import predict

MODEL_PATH = "app/model/modelo_lego_final.pth"
model = load_model(MODEL_PATH)

st.title("📸 Cargar Imagen para Identificación")

uploaded_file = st.file_uploader("Sube una imagen del set de LEGO", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    predicted_class, confidence = predict(image, model)
    st.subheader(f"🔍 Set identificado: {predicted_class}")
    st.write(f"📈 Confianza: {confidence:.2f}%")
