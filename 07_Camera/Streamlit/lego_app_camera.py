import os
import streamlit as st

st.write("📂 Directorio actual:", os.getcwd())
st.write("📂 Archivos disponibles:", os.listdir(os.getcwd()))
