import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import pymongo
import psycopg2

# Configuración de la app
st.set_page_config(page_title="Ironbrick", page_icon="🧱", layout="wide")

# Sidebar para la navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Selecciona una página", ["Recomendador de Inversión", "Alertas de Telegram"])

#ok
# Conexión a MongoDB
@st.cache_resource
def init_mongo_connection():
    return pymongo.MongoClient(st.secrets["mongo"]["uri"])

mongo_client = init_mongo_connection()
mongo_db = mongo_client[st.secrets["mongo"]["db"]]
mongo_collection = mongo_db[st.secrets["mongo"]["collection"]]

#ok
# Conexión a PostgreSQL
DB_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DB_URL, sslmode="require")

#ok
# Cargar modelo de predicción
modelo_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/05_Streamlit/models/stacking_model.pkl"

@st.cache_resource
def load_model():
    modelo_path = "/tmp/stacking_model.pkl"
    if not os.path.exists(modelo_path):
        response = requests.get(modelo_url)
        with open(modelo_path, "wb") as f:
            f.write(response.content)
    return joblib.load(modelo_path)

modelo = load_model()

#ok
# Cargar datos desde MongoDB
@st.cache_data(ttl=600)
def load_data():
    data = list(mongo_collection.find({}, {"_id": 0}))
    if not data:
        st.error("❌ No se encontraron datos en MongoDB.")
        st.stop()
    df = pd.DataFrame(data)
    return preprocess_data(df)

def preprocess_data(df):
    df = df[df['USRetailPrice'] > 0].copy()

    if 'Exclusivity' in df.columns:
        exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
        df['Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)

    if 'SizeCategory' in df.columns:
        size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
        df['SizeCategory'] = df['SizeCategory'].map(size_category_mapping)

    # Asegurar la creación de columnas faltantes
    df["PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]
    df["PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)
    df["YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]

    # Llenar valores NaN o Inf con 0
    df.replace([np.inf, -np.inf], 0, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

df_lego = load_data()

# if page == "Recomendador de Inversión":
#     st.title("Recomendador de sets actuales para Inversión en LEGO 📊")

#     st.write("**Explicación:** Según el presupuesto y los temas de interés seleccionados, el sistema generará un ranking de los 3 sets más rentables para invertir en LEGO. Se ha entrenado un modelo de Machine Learning que predice la rentabilidad de un set en los próximos años, basado en características como el precio, el número de piezas, la exclusividad, etc.")

#     st.markdown("""
#     ### Código de Color para Evaluación de Riesgo:        """)
#     st.write("**Todos los sets recomendados tienen una alta rentabilidad basada en sus características.**. Hemos analizado el riesgo y  clasificado con una escala de color:")
#     st.markdown("""
#     - 🟢 **Verde**: Set con una alta probabilidad de revalorización y rentabilidad.
#     - 🟡 **Amarillo**: Set con potencial de revalorización y con un riesgo medio.
#     - 🟠 **Naranja**: Set posibilidades de bajas de rentabilidad pero con riesgo medio-bajo
#     - 🔴 **Rojo**: Set con posibilidades de revalorización pero con una baja rentabilidad.
#     """)

#     st.subheader("Configura tu Inversión en LEGO")

#     # 📌 Configuración de presupuesto y temas
#     presupuesto_min, presupuesto_max = st.slider("💰 Rango de presupuesto (USD)", 10, 1000, (10, 200), step=10)

#     temas_unicos = sorted(df_lego["Theme"].unique().tolist())
#     temas_opciones = ["Todos"] + temas_unicos
#     selected_themes = st.multiselect("🛒 Selecciona los Themes de Interés", temas_opciones, default=["Todos"])

#     # 📌 Filtrar por presupuesto y temas
#     df_filtrado = df_lego[(df_lego["USRetailPrice"] >= presupuesto_min) & (df_lego["USRetailPrice"] <= presupuesto_max)]

#     if "Todos" not in selected_themes:
#         df_filtrado = df_filtrado[df_filtrado["Theme"].isin(selected_themes)]

#     # 📌 Si `df_filtrado` está vacío, mostrar error y detener ejecución
#     if df_filtrado.empty:
#         st.error("❌ No hay sets disponibles con los filtros seleccionados.")
#         st.stop()

#     # 📌 Funciones auxiliares para obtener imágenes y colores
#     def get_lego_image(set_number):
#         return f"https://img.bricklink.com/ItemImage/SN/0/{set_number}-1.png"


#     def get_color(score):
#         if score > 12:
#             return "#00736d"  # Verde
#         elif score > 6:
#             return "#FFC300"  # Amarillo
#         elif score > 2:
#             return "#FF9944"  # Naranja
#         else:
#             return "#FF4B4B"  # Rojo

#     # 📌 Generar Predicciones y Mostrar Top 3 Sets
#     if st.button("Generar Predicciones"):
#         features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit',
#                     'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity',
#                     'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']
#         df_filtrado["PredictedInvestmentScore"] = modelo.predict(df_filtrado[features])
#         df_filtrado = df_filtrado[df_filtrado["PredictedInvestmentScore"] > 0].sort_values(by="PredictedInvestmentScore", ascending=False).head(3)
#         st.subheader("📊 Top 3 Sets Más Rentables")
#         st.write(df_filtrado[["SetName", "Theme", "USRetailPrice", "PredictedInvestmentScore"]])

# elif page == "Alertas de Telegram":
#     st.title("📢 Configuración de Alertas de Telegram")

#     telegram_id = st.text_input("🔹 Tu ID de Telegram (@userinfobot)")
#     presupuesto_min, presupuesto_max = st.slider("💰 Rango de presupuesto (USD)", 10, 500, (10, 200), step=10)

#     temas_unicos = sorted(df_lego["Theme"].unique().tolist())
#     temas_opciones = ["Todos"] + temas_unicos
#     temas_favoritos = st.multiselect("🛒 Temas Favoritos", temas_opciones, default=["Todos"])

#     if st.button("💾 Alta en Alertas"):
#         temas_str = ",".join(temas_favoritos)
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         cursor.execute("""
#         CREATE TABLE IF NOT EXISTS usuarios (
#             telegram_id TEXT PRIMARY KEY,
#             presupuesto_min INTEGER,
#             presupuesto_max INTEGER,
#             temas_favoritos TEXT
#         )""")

#         cursor.execute("""
#         INSERT INTO usuarios (telegram_id, presupuesto_min, presupuesto_max, temas_favoritos)
#         VALUES (%s, %s, %s, %s)
#         ON CONFLICT (telegram_id) DO UPDATE
#         SET presupuesto_min = EXCLUDED.presupuesto_min,
#             presupuesto_max = EXCLUDED.presupuesto_max,
#             temas_favoritos = EXCLUDED.temas_favoritos;
#         """, (telegram_id, presupuesto_min, presupuesto_max, temas_str))

#         conn.commit()
#         conn.close()
#         st.success("✅ Preferencias guardadas correctamente!")


#     features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit',
#             'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity',
#             'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

#     df_lego["PredictedInvestmentScore"] = modelo.predict(df_lego[features])

#     # Transformamos los valores de revalorización en categorías
#     def clasificar_revalorizacion(score):
#         if score > 13:
#             return "Muy Alta"
#         elif 10 <= score <= 13:
#             return "Alta"
#         elif 5 <= score < 10:
#             return "Media"
#         elif 0 <= score < 5:
#             return "Baja"
#         else:
#             return "Ninguna"

#     df_lego["Revalorización"] = df_lego["PredictedInvestmentScore"].apply(clasificar_revalorizacion)

#     df_lego.rename(columns={
#         "Number": "Set",
#         "SetName": "Nombre",
#         "USRetailPrice": "Precio",
#         "Theme": "Tema"
#     }, inplace=True)

#     st.write("📊 **Sets Recomendados por IronbrickML**:")
#     df_recomendados = df_lego[df_lego["PredictedInvestmentScore"] > 0].sort_values(by="PredictedInvestmentScore", ascending=False)
#     st.data_editor(df_recomendados[["Set", "Nombre", "Precio", "Tema", "Revalorización"]], disabled=True)

#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT telegram_id, presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios")
#     usuarios = cursor.fetchall()
#     conn.close()

#     if usuarios:
#         df_usuarios = pd.DataFrame(usuarios, columns=["Telegram ID", "Presupuesto Mín", "Presupuesto Máx", "Temas Favoritos"])
#         st.dataframe(df_usuarios)
#     else:
#         st.warning("❌ No hay usuarios registrados.")
