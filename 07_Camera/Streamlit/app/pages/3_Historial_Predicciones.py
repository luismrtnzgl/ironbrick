import streamlit as st
import pandas as pd

# 📌 Cargar historial de predicciones (archivo CSV)
HISTORIAL_PATH = "app/historial_predicciones.csv"

def load_historial():
    try:
        return pd.read_csv(HISTORIAL_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Fecha", "Set Predicho", "Confianza"])

st.title("📊 Historial de Predicciones")

historial = load_historial()
st.dataframe(historial)

st.download_button(
    label="📥 Descargar Historial",
    data=historial.to_csv(index=False).encode("utf-8"),
    file_name="historial_predicciones.csv",
    mime="text/csv"
)
