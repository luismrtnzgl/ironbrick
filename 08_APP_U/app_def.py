import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import pymongo
import psycopg2
import pickle
import itertools
import matplotlib.pyplot as plt
import torch
import urllib.request
import json
import asyncio
from model_utils import load_model
from predict import predict
from streamlit_option_menu import option_menu
from base64 import b64encode

# Configuración de la app
st.set_page_config(page_title="Ironbrick", page_icon="08_APP_U/ironbrick.ico", layout="wide")


# Inicializar session_state
if "page" not in st.session_state:
    st.session_state.page = "Recomendador de Inversión en sets Actuales"


# Sidebar para la navegación
image = "08_APP_U/logo_ironbrick.jpg"
st.sidebar.image(image,use_container_width=True)
#page = st.sidebar.radio("Selecciona una página", ["Recomendador de Inversión", "Alertas de Telegram"])

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stButton]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:

    app = option_menu(
        menu_title="",
        options=["Inicio","Recomendador de Inversión en sets Actuales", "Recomendador de Inversión en sets Retirados","Alertas de Telegram", "Identificador de Sets"],
        icons=["📌", "📌", "📌", "📌"],
        styles = {
            "container": {"background-color": "#ffef47", "margin-top": "10px", "padding": "8px"},
            "nav-link": {"background-color": "#92cfce", "color": "black"},
            "nav-link-selected": {"background-color": "#e87577", "color": "white"}

        }
    )
#opcion 2 menu sidebar
# # Contenedor de botones centrados
# with st.sidebar.container():
#     st.markdown('<div class="button-container">', unsafe_allow_html=True)

#     if st.button("Recomendador de Inversión en sets Actuales"):
#         st.session_state.page = "Recomendador de Inversión en sets Actuales"

#     if st.button("Recomendador de Inversión en sets Retirados"):
#         st.session_state.page = "Recomendador de Inversión en sets Retirados"

#     if st.button("Alertas de Telegram"):
#         st.session_state.page = "Alertas de Telegram"

#     if st.button("Identificador de Sets"):
#         st.session_state.page = "Identificador de Sets"

#     st.markdown('</div>', unsafe_allow_html=True)






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


# ✅ Página principal por defecto
if not app:
    st.title("Ironbrick")
    st.write("Bienvenido a IA for Lego Investments.")
    st.write("Selecciona una opción del menú lateral para comenzar.")





# ✅ Muestra la página seleccionada
#if st.session_state.page == "Recomendador de Inversión en sets Actuales": #opcion menu 2
# Controlar qué página mostrar según la opción seleccionada
elif app == "Recomendador de Inversión en sets Actuales":

    # Abrir la imagen en modo binario
    with open("08_APP_U/IRONBRICK_APP_1_PEQ.png", "rb") as img_file:
        img_data = b64encode(img_file.read()).decode("utf-8")  # Codificar la imagen en base64

  # Usar HTML para mostrar la imagen y el texto a la derecha
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{img_data}" alt="lego" style="width: 150px; height: 150px; margin-right: 20px;">
            <h3 style="margin: 0;">Recomendador de sets actuales para Inversión en LEGO</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("**Explicación:** Según el presupuesto y los temas de interés seleccionados, el sistema generará un ranking de los 3 sets más rentables para invertir en LEGO. Se ha entrenado un modelo de Machine Learning que predice la rentabilidad de un set en los próximos años, basado en características como el precio, el número de piezas, la exclusividad, etc.")

    st.markdown("""
    ### Código de Color para Evaluación de Riesgo:        """)
    st.write("**Todos los sets recomendados tienen una alta rentabilidad basada en sus características.**. Hemos analizado el riesgo y  clasificado con una escala de color:")
    st.markdown("""
    - 🟢 **Verde**: Set con una alta probabilidad de revalorización y rentabilidad.
    - 🟡 **Amarillo**: Set con potencial de revalorización y con un riesgo medio.
    - 🟠 **Naranja**: Set posibilidades de bajas de rentabilidad pero con riesgo medio-bajo
    - 🔴 **Rojo**: Set con posibilidades de revalorización pero con una baja rentabilidad.
    """)

    st.subheader("Configura tu Inversión en LEGO")

    # 📌 Configuración de presupuesto y temas
    presupuesto_min, presupuesto_max = st.slider("💰 Rango de presupuesto (USD)", 10, 1000, (10, 200), step=10)

    temas_unicos = sorted(df_lego["Theme"].unique().tolist())
    temas_opciones = ["Todos"] + temas_unicos
    selected_themes = st.multiselect("🛒 Selecciona los Themes de Interés", temas_opciones, default=["Todos"])

    # 📌 Filtrar por presupuesto y temas
    df_filtrado = df_lego[(df_lego["USRetailPrice"] >= presupuesto_min) & (df_lego["USRetailPrice"] <= presupuesto_max)]

    if "Todos" not in selected_themes:
        df_filtrado = df_filtrado[df_filtrado["Theme"].isin(selected_themes)]

    # 📌 Si `df_filtrado` está vacío, mostrar error y detener ejecución
    if df_filtrado.empty:
        st.error("❌ No hay sets disponibles con los filtros seleccionados.")
        st.stop()

    # 📌 Funciones auxiliares para obtener imágenes y colores
    def get_lego_image(set_number):
        return f"https://img.bricklink.com/ItemImage/SN/0/{set_number}-1.png"


    def get_color(score):
        if score > 12:
            return "#00736d"  # Verde
        elif score > 6:
            return "#FFC300"  # Amarillo
        elif score > 2:
            return "#FF9944"  # Naranja
        else:
            return "#FF4B4B"  # Rojo

    # # 📌 Generar Predicciones y Mostrar Top 3 Sets
    if st.button("Generar Predicciones"):
        if "PredictedInvestmentScore" not in df_filtrado.columns:
            features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit',
                        'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity',
                        'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

#         # 📌 Asegurar que hay datos antes de predecir
            if df_filtrado.shape[0] == 0:
                st.error("❌ No hay sets disponibles para predecir. Prueba ajustando los filtros.")
                st.stop()

            df_filtrado = df_filtrado.copy()
            df_filtrado.loc[:, "PredictedInvestmentScore"] = modelo.predict(df_filtrado[features])
            df_filtrado = df_filtrado[df_filtrado["PredictedInvestmentScore"] > 0]

            if df_filtrado.shape[0] < 3:
                st.warning("⚠️ Menos de 3 sets cumplen con los criterios seleccionados. Mostrando los disponibles.")

            df_filtrado = df_filtrado.sort_values(by="PredictedInvestmentScore", ascending=False).head(3)

            st.subheader("📊 Top 3 Sets Más Rentables")
            if not df_filtrado.empty:
                cols = st.columns(len(df_filtrado))
                for col, (_, row) in zip(cols, df_filtrado.iterrows()):
                    with col:
                        color = get_color(row["PredictedInvestmentScore"])
                        st.markdown(f"""
                            <div style='background-color:{color}; padding:10px; border-radius:5px; text-align:center; margin-bottom:10px;'>
                                <strong>{row['SetName']}</strong>
                            </div>
                        """, unsafe_allow_html=True)
                        image_url = get_lego_image(row["Number"])
                        st.image(image_url, caption=row["SetName"], use_container_width=True)
                        st.write(f"**Tema:** {row['Theme']}")
                        st.write(f"💰 **Precio:** ${row['USRetailPrice']:.2f}")
                        url_lego = f"https://www.lego.com/en-us/product/{row['Number']}"
                        st.markdown(f'<a href="{url_lego}" target="_blank"><button style="background-color:#ff4b4b; border:none; padding:10px; border-radius:5px; cursor:pointer; font-size:14px;">🛒 Comprar en LEGO</button></a>', unsafe_allow_html=True)
                        st.write("---")

# ✅ Muestra la página seleccionada
#if st.session_state.page == "Recomendador de Inversión en sets Retirados":

elif app == "Recomendador de Inversión en sets Retirados":
    # Obtenemos la ruta del archivo CSV
    BASE_DIR = os.getcwd()
    CSV_PATH = os.path.join(BASE_DIR, "04_Extra/APP/data/scraped_lego_data.csv")

    # Verificamos si el archivo existe
    if not os.path.exists(CSV_PATH):
        st.error("❌ ERROR: El archivo CSV NO EXISTE en la ruta especificada.")
        st.stop()

    # Cargamos el archivo CSV
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        st.error(f"❌ ERROR al leer el archivo CSV: {e}")
        st.stop()

    # Procesamos el dataset
    df["PriceDate"] = pd.to_datetime(df["PriceDate"], errors='coerce')
    df = df.dropna(subset=["PriceDate"])
    df_sorted = df.sort_values(by=['Number', 'PriceDate'])
    df_sorted['PriceIndex'] = df_sorted.groupby('Number').cumcount()
    df_transformed = df_sorted.pivot(index=['Number', 'SetName', 'Theme', 'Year', 'Pieces',
                                            'RetailPriceUSD', 'CurrentValueNew', 'ForecastValueNew2Y',
                                            'ForecastValueNew5Y'],
                                    columns='PriceIndex', values='PriceValue').reset_index()
    df_transformed.columns = [f'Price_{col+1}' if isinstance(col, int) else col for col in df_transformed.columns]
    price_columns = [f'Price_{i}' for i in range(1, 13)]
    df_transformed = df_transformed[['Number', 'SetName', 'Theme', 'Year', 'Pieces', 'RetailPriceUSD', 'CurrentValueNew', 'ForecastValueNew2Y', 'ForecastValueNew5Y'] + price_columns]
    df_transformed[price_columns] = df_transformed[price_columns].fillna(0)
    df_transformed.loc[:, 'Pieces'] = df_transformed['Pieces'].fillna(0)
    df_transformed.loc[:, 'RetailPriceUSD'] = df_transformed['RetailPriceUSD'].fillna(0)
    df_transformed.loc[df_transformed['CurrentValueNew'] == 0, 'CurrentValueNew'] = df_transformed['RetailPriceUSD']
    df_transformed = df_transformed.dropna()

    # Cargamos modelos de predicción
    pkl_path_2y = os.path.join(BASE_DIR, "04_Extra/APP/models/xgb_2y.pkl")
    pkl_path_5y = os.path.join(BASE_DIR, "04_Extra/APP/models/xgb_5y.pkl")

    if not os.path.exists(pkl_path_2y) or not os.path.exists(pkl_path_5y):
        st.error("❌ No se encontraron los modelos .pkl en la carpeta 'models/'.")
        st.stop()

    @st.cache_resource
    def load_model(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

    model_2y = load_model(pkl_path_2y)
    model_5y = load_model(pkl_path_5y)

    # Generamos predicciones
    df_identification = df_transformed[['Number', 'SetName', 'Theme', 'CurrentValueNew']].copy()
    df_model = df_transformed.drop(columns=['Number', 'SetName', 'Theme'], errors='ignore').copy()
    df_model = pd.get_dummies(df_model, drop_first=True)
    expected_columns = model_2y.feature_names_in_
    for col in expected_columns:
        if col not in df_model.columns:
            df_model[col] = 0
    df_model = df_model[expected_columns]
    df_identification.loc[:, 'PredictedValue2Y'] = model_2y.predict(df_model)
    df_identification.loc[:, 'PredictedValue5Y'] = model_5y.predict(df_model)

    # Calculamos rentabilidad porcentual por tema
    df_identification["Rentabilidad2Y"] = ((df_identification["PredictedValue2Y"] - df_identification["CurrentValueNew"]) / df_identification["CurrentValueNew"]) * 100
    df_identification["Rentabilidad5Y"] = ((df_identification["PredictedValue5Y"] - df_identification["CurrentValueNew"]) / df_identification["CurrentValueNew"]) * 100

    df_rentabilidad_temas = df_identification.groupby("Theme").agg(
        TotalSets=('Theme', 'count'),
        Rentabilidad2Y=('Rentabilidad2Y', 'mean'),
        Rentabilidad5Y=('Rentabilidad5Y', 'mean')
    ).reset_index()

    df_rentabilidad_temas = df_rentabilidad_temas.sort_values(by="Rentabilidad5Y", ascending=False)

    # Abrir la imagen en modo binario
    with open("08_APP_U/IRONBRICK_APP_2_PEQ.png", "rb") as img_file:
        img_data = b64encode(img_file.read()).decode("utf-8")  # Codificar la imagen en base64

  # Usar HTML para mostrar la imagen y el texto a la derecha
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{img_data}" alt="lego" style="width: 150px; height: 150px; margin-right: 20px;">
            <h3 style="margin: 0;">Recomendador de inversión en sets de LEGO retirados</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("Este recomendador te ayuda a encontrar las mejores combinaciones de sets de LEGO retirados para invertir, basándose en su rentabilidad futura.")

    # Mostramos rentabilidad media porcentual por tema con total de sets
    st.subheader("📊 Rentabilidad media porcentual por tema")
    st.write("Este cuadro muestra la rentabilidad media estimada de los sets nuevos en 2 y 5 años para cada tema de LEGO, junto con el número total de sets evaluados en cada tema.")

    st.dataframe(
        df_rentabilidad_temas.style.format({"Rentabilidad a 2 años": "{:.2f}%", "Rentabilidad a 5 años": "{:.2f}%", "Total Sets": "{:.0f}"}),
        height=250,  # Altura ajustable
        use_container_width=True  # Tabla ocupe todo el ancho
    )

    # Selección de temas
    st.subheader("🎯 Selecciona tus temas de interés")
    temas_disponibles = sorted(df_identification["Theme"].unique())
    temas_seleccionados = st.multiselect("Selecciona los temas de interés", ["Todos"] + temas_disponibles, default=["Todos"])

    if "Todos" in temas_seleccionados:
        temas_seleccionados = temas_disponibles

    # Filtramos por presupuesto
    presupuesto = st.slider("Presupuesto máximo ($)", min_value=100, max_value=2000, value=200, step=10)

    # Filtramos el dataframe
    df_filtrado = df_identification[df_identification["Theme"].isin(temas_seleccionados)]
    df_filtrado = df_filtrado[df_filtrado["CurrentValueNew"] <= presupuesto]

    # Optimización: Seleccionamos los 10 mejores sets primero
    df_top_sets = df_filtrado.sort_values(by="PredictedValue5Y", ascending=False).head(10)

    # Buscamos combinaciones óptimas de inversión
    def encontrar_mejores_inversiones(df, presupuesto, num_opciones=3):
        sets_lista = df[['SetName', 'CurrentValueNew', 'PredictedValue2Y', 'PredictedValue5Y']].values.tolist()
        mejores_combinaciones = []

        for r in range(1, 5):  # Limitamos combinaciones a 1-4 sets para optimizar
            for combinacion in itertools.combinations(sets_lista, r):
                total_precio = sum(item[1] for item in combinacion)
                retorno_2y = sum(item[2] for item in combinacion)
                retorno_5y = sum(item[3] for item in combinacion)

                if total_precio <= presupuesto:
                    mejores_combinaciones.append((combinacion, retorno_2y, retorno_5y, total_precio))

        # Ordenamos por rentabilidad en 5 años (descendente)
        mejores_combinaciones.sort(key=lambda x: x[2], reverse=True)
        return mejores_combinaciones[:num_opciones]


    # Mostramos inversiones óptimas con imágenes y texto centrado
    if st.button("🔍 Buscar inversiones óptimas"):
        opciones = encontrar_mejores_inversiones(df_top_sets, presupuesto)

        if not opciones:
            st.warning("⚠️ No se encontraron combinaciones dentro de tu presupuesto.")
        else:
            st.subheader("💡 Mejores opciones de inversión")
            for i, (combo, ret_2y, ret_5y, precio) in enumerate(opciones, 1):
                st.write(f"**Opción {i}:**")
                st.write(f"💵 **Total de la inversión:** ${precio:.2f}")
                st.write(f"📈 **Valor estimado en 2 años:** ${ret_2y:.2f}")
                st.write(f"🚀 **Valor estimado en 5 años:** ${ret_5y:.2f}")
                st.write("🧩 **Sets incluidos:**")

                # Mostramos sets con imágenes y datos centrados
                cols = st.columns(len(combo))  # Crear columnas dinámicas para mostrar imágenes
                for col, (set_name, price, _, _) in zip(cols, combo):
                    set_number = df_top_sets[df_top_sets["SetName"] == set_name]["Number"].values[0]  # Obtener el número del set
                    image_url = f"https://images.brickset.com/sets/images/{set_number}.jpg"

                    with col:
                        st.image(image_url, use_container_width=True)  # Mostrar la imagen
                        st.markdown(
                            f"""
                            <div style="text-align: center;">
                                <strong>{set_name}</strong><br>
                                📌 <strong>Set {set_number}</strong><br>
                                💵 <strong>${price:.2f}</strong>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                st.write("---")



# ✅ Muestra la página seleccionada
#if st.session_state.page == "Alertas de Telegram":
elif app == "Alertas de Telegram":

     # Abrir la imagen en modo binario
    with open("08_APP_U/IRONBRICK_APP_3_PEQ.png", "rb") as img_file:
        img_data = b64encode(img_file.read()).decode("utf-8")  # Codificar la imagen en base64

  # Usar HTML para mostrar la imagen y el texto a la derecha
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{img_data}" alt="lego" style="width: 150px; height: 150px; margin-right: 20px;">
            <h3 style="margin: 0;">Alerta mensual de Inversión en LEGO por Telegram</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("**Bienvenido a IronbrickML - Alertas de Inversión en LEGO**")
    st.write(
        "📊 IronbrickML analiza la rentabilidad de sets de LEGO utilizando modelos de predicción de inversión. "
        "Cada mes, recibirás una recomendación personalizada en Telegram con el set que mejor se ajuste a tu presupuesto y preferencias. "
        "Solo se te sugerirán sets con alto potencial de revalorización y sin repeticiones para que siempre tengas nuevas oportunidades de inversión. "
        "Configura tus preferencias y deja que la inteligencia artificial haga el resto."
    )

    telegram_id = st.text_input("🔹 Tu ID de Telegram (usa @userinfobot en Telegram para obtenerlo)")
    presupuesto_min, presupuesto_max = st.slider("💰 Rango de presupuesto (USD)", 10, 500, (10, 200), step=10)

    temas_unicos = sorted(df_lego["Theme"].unique().tolist())
    temas_opciones = ["Todos"] + temas_unicos
    temas_favoritos = st.multiselect("🛒 Temas Favoritos", temas_opciones, default=["Todos"])

    if st.button("💾 Alta en Alertas"):
        temas_str = ",".join(temas_favoritos)
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            telegram_id TEXT PRIMARY KEY,
            presupuesto_min INTEGER,
            presupuesto_max INTEGER,
            temas_favoritos TEXT
        )""")

        cursor.execute("""
        INSERT INTO usuarios (telegram_id, presupuesto_min, presupuesto_max, temas_favoritos)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (telegram_id) DO UPDATE
        SET presupuesto_min = EXCLUDED.presupuesto_min,
            presupuesto_max = EXCLUDED.presupuesto_max,
            temas_favoritos = EXCLUDED.temas_favoritos;
        """, (telegram_id, presupuesto_min, presupuesto_max, temas_str))

        conn.commit()
        conn.close()
        st.success("✅ Preferencias guardadas correctamente!")


    features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit',
            'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity',
            'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

    df_lego["PredictedInvestmentScore"] = modelo.predict(df_lego[features])

    # Transformamos los valores de revalorización en categorías
    def clasificar_revalorizacion(score):
        if score > 13:
            return "Muy Alta"
        elif 10 <= score <= 13:
            return "Alta"
        elif 5 <= score < 10:
            return "Media"
        elif 0 <= score < 5:
            return "Baja"
        else:
            return "Ninguna"

    df_lego["Revalorización"] = df_lego["PredictedInvestmentScore"].apply(clasificar_revalorizacion)

    df_lego.rename(columns={
        "Number": "Set",
        "SetName": "Nombre",
        "USRetailPrice": "Precio",
        "Theme": "Tema"
    }, inplace=True)

    st.write("📊 **Sets Recomendados por IronbrickML**:")
    df_recomendados = df_lego[df_lego["PredictedInvestmentScore"] > 0].sort_values(by="PredictedInvestmentScore", ascending=False)
    st.data_editor(df_recomendados[["Set", "Nombre", "Precio", "Tema", "Revalorización"]], disabled=True)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT telegram_id, presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios")
    usuarios = cursor.fetchall()


    if usuarios:
        df_usuarios = pd.DataFrame(usuarios, columns=["Telegram ID", "Presupuesto Mín", "Presupuesto Máx", "Temas Favoritos"])
        st.dataframe(df_usuarios)
    else:
        st.warning("❌ No hay usuarios registrados.")

    conn.close()

# ✅ Muestra la página seleccionada
#if st.session_state.page == "Identificador de Sets":
elif app == "Identificador de Sets":

    # Intentamos solucionar el error "no running event loop" en Streamlit
    if not hasattr(asyncio, "WindowsSelectorEventLoopPolicy") and os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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

    from model_utils import load_model

    # Cargamos el modelo entrenado correctamente
    try:
        model = load_model(MODEL_PATH)  # Ahora usa la versión correcta de `load_model`
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        model = None

    # Cargamos el mapeo de clases
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


         # Abrir la imagen en modo binario
    with open("08_APP_U/IRONBRICK_APP_4_PEQ.png", "rb") as img_file:
        img_data = b64encode(img_file.read()).decode("utf-8")  # Codificar la imagen en base64

  # Usar HTML para mostrar la imagen y el texto a la derecha
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{img_data}" alt="lego" style="width: 150px; height: 150px; margin-right: 20px;">
            <h3 style="margin: 0;">Identificación de Sets LEGO</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

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
                        st.image(image_url, caption=f"Set: {predicted_set_number}", width=300)
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
