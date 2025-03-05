import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os

# 📌 URL del modelo en GitHub RAW
modelo_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/05_Streamlit/models/stacking_model.pkl"

@st.cache_resource
def load_model():
    """Descarga el modelo desde GitHub y lo carga en Streamlit."""
    modelo_path = "/tmp/stacking_model.pkl"
    
    if not os.path.exists(modelo_path):
        response = requests.get(modelo_url)
        with open(modelo_path, "wb") as f:
            f.write(response.content)
    
    return joblib.load(modelo_path)

# 📌 Cargar el modelo
modelo = load_model()

# 📌 URL del dataset en GitHub RAW
dataset_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/01_Data_Cleaning/df_lego_final_venta.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(dataset_url)
    return preprocess_data(df)  # Aplicar preprocesamiento antes de usarlo

# 📌 Función de preprocesamiento (igual que en telegram_app.py)
def preprocess_data(df):
    df = df[df['USRetailPrice'] > 0].copy()

    exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
    df['Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)

    size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
    df['SizeCategory'] = df['SizeCategory'].map(size_category_mapping)

    df["PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]
    df["PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)
    df["YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

# 📌 Cargar dataset con preprocesamiento
df_ranking = load_data()

# 📌 Aplicar el modelo para predecir rentabilidad en TODOS los sets
features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 
            'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity', 
            'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

df_ranking["PredictedInvestmentScore"] = modelo.predict(df_ranking[features])

# 📌 Interfaz en Streamlit
st.title("Recomendador de sets actuales para Inversión en LEGO 📊")

st.write("**Explicación:** Según el presupuesto y los temas de interés seleccionados, el sistema generará un ranking de los sets más rentables para invertir en LEGO.")

# 📌 Configuración de presupuesto y temas
presupuesto_min, presupuesto_max = st.slider("💰 Selecciona el rango de presupuesto (USD)", 10, 500, (10, 200), step=10)

temas_unicos = sorted(df_ranking["Theme"].unique().tolist())
temas_opciones = ["Todos"] + temas_unicos
selected_themes = st.multiselect("🛒 Selecciona los Themes de Interés", temas_opciones, default=["Todos"])

# 📌 Filtrar por presupuesto y temas
df_filtrado = df_ranking[(df_ranking["USRetailPrice"] >= presupuesto_min) & 
                          (df_ranking["USRetailPrice"] <= presupuesto_max)]

if "Todos" not in selected_themes:
    df_filtrado = df_filtrado[df_filtrado["Theme"].isin(selected_themes)]

# 📌 Seleccionar columnas y renombrarlas
df_filtrado = df_filtrado[["Number", "Theme", "SetName", "USRetailPrice", "WantCount", "PredictedInvestmentScore"]].copy()

df_filtrado.rename(columns={
    "Number": "ID",
    "Theme": "Tema",
    "SetName": "Nombre del set",
    "USRetailPrice": "Precio de compra",
    "WantCount": "Personas que lo quieren",
    "PredictedInvestmentScore": "Rentabilidad como inversión"
}, inplace=True)

# 📌 Guardar la predicción numérica original en una nueva columna
df_filtrado["Score Numérico"] = df_filtrado["Rentabilidad como inversión"]

# 📌 Convertir la rentabilidad en categorías de texto
def clasificar_rentabilidad(score):
    if score > 10:
        return "Alta"
    elif 5 <= score <= 10:
        return "Media"
    else:
        return "Baja"

df_filtrado["Rentabilidad como inversión"] = df_filtrado["Score Numérico"].apply(clasificar_rentabilidad)

# 📌 Ordenar por rentabilidad de mayor a menor
df_filtrado = df_filtrado.sort_values(by="Score Numérico", ascending=False)

# 📌 Funciones auxiliares
def get_lego_image(set_number):
    return f"https://images.brickset.com/sets/images/{set_number}-1.jpg"

def get_color(score):
    if score > 12:
        return "#00736d"  # Verde
    elif score > 6:
        return "#FFC300"  # Amarillo
    elif score > 2:
        return "#FF9944"  # Naranja
    else:
        return "#FF4B4B"  # Rojo

# 📌 Mostrar los resultados en Streamlit
if st.button("Generar Predicciones"):
    st.subheader("📊 Sets Recomendados para Inversión")
    if not df_filtrado.empty:
        cols = st.columns(len(df_filtrado))
        for col, (_, row) in zip(cols, df_filtrado.iterrows()):
            with col:
                color = get_color(row["Score Numérico"])  # Usar la columna numérica
                st.markdown(f"""
                    <div style='background-color:{color}; padding:10px; border-radius:5px; text-align:center; margin-bottom:10px;'>
                        <strong>{row['Nombre del set']}</strong>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div style='display: flex; justify-content: center;'>
                        <img src='{get_lego_image(row["ID"])}' width='100%'>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
                st.write(f"**Tema:** {row['Tema']}")
                st.write(f"💰 **Precio:** ${row['Precio de compra']:.2f}")
                url_lego = f"https://www.lego.com/en-us/product/{row['ID']}"
                st.markdown(f'<a href="{url_lego}" target="_blank"><button style="background-color:#ff4b4b; border:none; padding:10px; border-radius:5px; cursor:pointer; font-size:14px;">🛒 Comprar en LEGO</button></a>', unsafe_allow_html=True)
                st.write("---")
    else:
        st.error("❌ No hay sets disponibles según los criterios seleccionados.")
