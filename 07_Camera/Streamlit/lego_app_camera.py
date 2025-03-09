import streamlit as st
import torch
import os
import urllib.request
import pandas as pd
import json
import asyncio
from model_utils import load_model
from predict import predict

# Intentamos solucionar el error "no running event loop" en Streamlit
if not hasattr(asyncio, "WindowsSelectorEventLoopPolicy") and os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

st.set_page_config(page_title="Identificación de Sets LEGO", layout="wide")

# Definimos las rutas de archivos locales
MODEL_PATH = "modelo_lego_final.pth"
MAPPING_PATH = "idx_to_class.json"
DATA_PATH = "df_lego_camera.csv"

# URLs de los archivos en GitHub (raw) para descargar
MODEL_URL = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/07_Camera/Streamlit/modelo_lego_final.pth"
MAPPING_URL = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/07_Camera/Streamlit/idx_to_class.json"
DATASET_URL = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/07_Camera/Streamlit/df_lego_camera.csv"

# Descargamos los archivos si no existen mostramos un error
def download_file(url, path):
    try:
        if not os.path.exists(path):
            st.write(f"📥 Descargando {path} desde GitHub...")
            urllib.request.urlretrieve(url, path)
            st.write(f"✅ {path} descargado exitosamente.")
    except Exception as e:
        st.error(f"❌ Error al descargar {path}: {e}")

download_file(MODEL_URL, MODEL_PATH)
download_file(MAPPING_URL, MAPPING_PATH)
download_file(DATASET_URL, DATA_PATH)

# Cargamos el modelo entrenado forzando la carga de CPU
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    model = None

# Cargamos el mapeo de clases de sets de LEGO
if os.path.exists(MAPPING_PATH):
    with open(MAPPING_PATH, "r") as f:
        idx_to_class = json.load(f)
else:
    st.error("❌ Error: No se encontró el archivo de mapeo idx_to_class.json.")
    idx_to_class = {}

# Cargamos el dataset actualizado con información de sets de LEGO convirtiendo el número del set a string
if os.path.exists(DATA_PATH):
    df_lego = pd.read_csv(DATA_PATH)
    df_lego["Number"] = df_lego["Number"].astype(str)
else:
    st.error("❌ Error: El archivo df_lego_camera.csv no se encontró.")
    df_lego = None

st.title("🧩 Identificación de Sets LEGO")

st.markdown(
    """
    🎯 **Sube una imagen o usa la cámara para identificar un set de LEGO.**
    - 📸 **Sube una imagen**
    - 📹 **Usa la cámara**
    """
)

# Definimos la captura de imagen
uploaded_file = st.file_uploader("Sube una imagen del set de LEGO", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    from PIL import Image
    image = Image.open(uploaded_file).convert("RGB")

    # Creamos dos columnas
    col1, col2 = st.columns([1, 1])

    # Columna 1: Mostramos la imagen cargada
    with col1:
        st.image(image, caption="Imagen subida", use_container_width=True)

    # Realizamos predicción en la Columna 2
    with col2:
        try:
            predicted_class, probabilities = predict(image, model)
            confidence = probabilities[0][predicted_class] * 100  # Convertir a porcentaje

            # Convertimos la predicción al número de set real
            predicted_set_number = str(idx_to_class.get(str(predicted_class), "Desconocido"))

            # Obtenemos la imagen desde la URL de Brickset
            image_url = f"https://images.brickset.com/sets/images/{predicted_set_number}-1.jpg"

            # Verificamos si la imagen existe en Brickset
            try:
                response = urllib.request.urlopen(image_url)
                if response.status == 200:
                    st.image(image_url, caption=f"Imagen de Brickset: {predicted_set_number}", width=300)
                else:
                    st.warning(f"⚠️ No se encontró imagen en Brickset para el set {predicted_set_number}.")
            except:
                st.warning(f"⚠️ No se encontró imagen en Brickset para el set {predicted_set_number}.")

            # Buscamos información en el dataset si está disponible
            if df_lego is not None:
                set_info = df_lego[df_lego["Number"] == predicted_set_number]

                if not set_info.empty:
                    set_name = set_info.iloc[0].get('SetName', 'Desconocido')
                    theme = set_info.iloc[0].get('Theme', 'Desconocido')
                    interested_people = set_info.iloc[0].get('WantCount', 'N/A')
                    retail_price = set_info.iloc[0].get('USRetailPrice', 'N/A')
                    used_price = set_info.iloc[0].get('BrickLinkSoldPriceUsed', 'N/A')

                    # Mostrar información del set
                    st.subheader(f"🔍 Set identificado: {set_name} ({predicted_set_number})")
                    st.write(f"🎭 **Tema:** {theme}")
                    st.write(f"👥 **Personas interesadas:** {interested_people}")
                    st.write(f"💰 **Precio original:** ${retail_price}")
                    st.write(f"🛒 **Precio usado:** ${used_price}")
                    st.write(f"📈 **Confianza en la predicción:** {confidence:.2f}%")
                else:
                    st.warning(f"⚠️ No se encontró información del set {predicted_set_number}.")
            else:
                st.error("❌ No se puede mostrar información del set porque el dataset no está disponible.")

        except Exception as e:
            st.error(f"❌ Error al hacer la predicción: {e}")

elif uploaded_file is not None and model is None:
    st.error("❌ No se puede hacer la predicción porque el modelo no se cargó correctamente.")