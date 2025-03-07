import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import pymongo



# Inicializa la conexión con MongoDB (se ejecuta solo una vez)
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(st.secrets["mongo"]["uri"])

client = init_connection()
db = client[st.secrets["mongo"]["db"]]  # Usar el nombre de la base de datos desde secrets.toml
collection = db[st.secrets["mongo"]["collection"]]

# 📌 URL del modelo en GitHub RAW
modelo_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/05_Streamlit/models/stacking_model.pkl"

# @st.cache_resource
def load_model():
    """Descarga el modelo desde GitHub y lo carga en Streamlit."""
    modelo_path = "/tmp/stacking_model.pkl"

    if not os.path.exists(modelo_path):
        response = requests.get(modelo_url)
        with open(modelo_path, "wb") as f:
            f.write(response.content)

    return joblib.load(modelo_path)


# # 📌 Cargar el modelo
modelo = load_model()



# 📌 Función para cargar datos desde MongoDB
@st.cache_data(ttl=600)
def load_data():
    data = list(collection.find({}, {"_id": 0}))  # Excluir `_id` para evitar problemas
    if not data:
        st.error("❌ No se encontraron datos en la colección de MongoDB.")
        st.stop()

    df = pd.DataFrame(data)
    st.write("📊 Datos cargados desde MongoDB:", df.head())  # Mostrar primeras filas
    return df


# 📌 Cargar dataset desde MongoDB
df_ranking = load_data()


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



# # 📌 Interfaz en Streamlit
st.title("Recomendador de sets actuales para Inversión en LEGO 📊")

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
presupuesto_min, presupuesto_max = st.slider("💰 Selecciona el rango de presupuesto (USD)", 10, 1000, (10, 200), step=10)


temas_unicos = sorted(df_ranking["Theme"].unique().tolist())
temas_opciones = ["Todos"] + temas_unicos
selected_themes = st.multiselect("🛒 Selecciona los Themes de Interés", temas_opciones, default=["Todos"])

# 📌 Filtrar por presupuesto y temas
df_filtrado = df_ranking[(df_ranking["USRetailPrice"] >= presupuesto_min) &
                          (df_ranking["USRetailPrice"] <= presupuesto_max)]

if "Todos" not in selected_themes:
    df_filtrado = df_filtrado[df_filtrado["Theme"].isin(selected_themes)]

# 📌 Si `df_filtrado` está vacío, mostrar error y detener ejecución
if df_filtrado.empty:
    st.error("❌ No hay sets disponibles con los filtros seleccionados. Ajusta la franja de precios o los temas para ver opciones.")
    st.stop()

# 📌 Funciones auxiliares para obtener imágenes y colores
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

# 📌 Generar Predicciones y Mostrar Top 3 Sets
if st.button("Generar Predicciones"):
    if "PredictedInvestmentScore" not in df_filtrado.columns:
        features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit',
                    'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity',
                    'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

        # 📌 Asegurar que hay datos antes de predecir
        if df_filtrado.shape[0] == 0:
            st.error("❌ No hay sets disponibles para predecir. Prueba ajustando los filtros.")
            st.stop()

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
                st.markdown(f"""
                    <div style='display: flex; justify-content: center;'>
                        <img src='{get_lego_image(row["Number"])}' width='100%'>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
                st.write(f"**Tema:** {row['Theme']}")
                st.write(f"💰 **Precio:** ${row['USRetailPrice']:.2f}")
                url_lego = f"https://www.lego.com/en-us/product/{row['Number']}"
                st.markdown(f'<a href="{url_lego}" target="_blank"><button style="background-color:#ff4b4b; border:none; padding:10px; border-radius:5px; cursor:pointer; font-size:14px;">🛒 Comprar en LEGO</button></a>', unsafe_allow_html=True)
                st.write("---")






















#inicio cambio erv
# 📌 Leer credenciales desde `secrets.toml`
# Accede a los secretos directamente
#mongo_uri = config("MONGO_URI")
#database_name = config("DATABASE_NAME")
#collection_name = config("COLLECTION_NAME")
#fin cambio erv





# 📌 URL del modelo en GitHub RAW
# modelo_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/05_Streamlit/models/stacking_model.pkl"

# @st.cache_resource
# def load_model():
#     """Descarga el modelo desde GitHub y lo carga en Streamlit."""
#     modelo_path = "/tmp/stacking_model.pkl"

#     if not os.path.exists(modelo_path):
#         response = requests.get(modelo_url)
#         with open(modelo_path, "wb") as f:
#             f.write(response.content)

#     return joblib.load(modelo_path)

# # 📌 Cargar el modelo
# modelo = load_model()

# @st.cache_data
# def load_data():
#     # Ya no necesitas reconectar a la base de datos, ya que la colección se importa desde config.py
#     data = list(collection.find())  # Usar la colección importada

#     if len(data) == 0:
#         st.error("❌ No se encontraron datos en la colección de MongoDB.")
#         st.stop()  # Detener la ejecución si no hay datos

#     df = pd.DataFrame(data)

#     # Verificar las primeras filas del DataFrame para asegurarse de que los datos están bien cargados
#     st.write("Datos cargados correctamente:", df.head())  # Mostrar las primeras filas del DataFrame
#     return df






#inicio original luis
# 📌 URL del dataset en GitHub RAW
#dataset_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/01_Data_Cleaning/df_lego_final_venta.csv"

#@st.cache_data
#def load_data():
    #df = pd.read_csv(dataset_url)
    #return preprocess_data(df)  # Aplicar preprocesamiento antes de usarlo
#fin original luis

#inicio original luis
# 📌 Función de preprocesamiento (igual que en telegram_app.py)
# def preprocess_data(df):
#     df = df[df['USRetailPrice'] > 0].copy()

#     exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
#     df['Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)

#     size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
#     df['SizeCategory'] = df['SizeCategory'].map(size_category_mapping)

#     df["PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]
#     df["PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)
#     df["YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]

#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

#     return df
#fin original Luis


# 📌 Cargar dataset con preprocesamiento
# df_ranking = load_data()
# #df_ranking = load_data_from_mongodb() #cambio erv

# # 📌 Interfaz en Streamlit
# st.title("Recomendador de sets actuales para Inversión en LEGO 📊")

# st.write("**Explicación:** Según el presupuesto y los temas de interés seleccionados, el sistema generará un ranking de los 3 sets más rentables para invertir en LEGO. Se ha entrenado un modelo de Machine Learning que predice la rentabilidad de un set en los próximos años, basado en características como el precio, el número de piezas, la exclusividad, etc.")

# st.markdown("""
# ### Código de Color para Evaluación de Riesgo:        """)
# st.write("**Todos los sets recomendados tienen una alta rentabilidad basada en sus características.**. Hemos analizado el riesgo y  clasificado con una escala de color:")
# st.markdown("""
# - 🟢 **Verde**: Set con una alta probabilidad de revalorización y rentabilidad.
# - 🟡 **Amarillo**: Set con potencial de revalorización y con un riesgo medio.
# - 🟠 **Naranja**: Set posibilidades de bajas de rentabilidad pero con riesgo medio-bajo
# - 🔴 **Rojo**: Set con posibilidades de revalorización pero con una baja rentabilidad.
# """)

# st.subheader("Configura tu Inversión en LEGO")

# # 📌 Configuración de presupuesto y temas
# presupuesto_min, presupuesto_max = st.slider("💰 Selecciona el rango de presupuesto (USD)", 10, 1000, (10, 200), step=10)


# temas_unicos = sorted(df_ranking["Theme"].unique().tolist())
# temas_opciones = ["Todos"] + temas_unicos
# selected_themes = st.multiselect("🛒 Selecciona los Themes de Interés", temas_opciones, default=["Todos"])

# # 📌 Filtrar por presupuesto y temas
# df_filtrado = df_ranking[(df_ranking["USRetailPrice"] >= presupuesto_min) &
#                           (df_ranking["USRetailPrice"] <= presupuesto_max)]

# if "Todos" not in selected_themes:
#     df_filtrado = df_filtrado[df_filtrado["Theme"].isin(selected_themes)]

# # 📌 Si `df_filtrado` está vacío, mostrar error y detener ejecución
# if df_filtrado.empty:
#     st.error("❌ No hay sets disponibles con los filtros seleccionados. Ajusta la franja de precios o los temas para ver opciones.")
#     st.stop()

# # 📌 Funciones auxiliares para obtener imágenes y colores
# def get_lego_image(set_number):
#     return f"https://images.brickset.com/sets/images/{set_number}-1.jpg"

# def get_color(score):
#     if score > 12:
#         return "#00736d"  # Verde
#     elif score > 6:
#         return "#FFC300"  # Amarillo
#     elif score > 2:
#         return "#FF9944"  # Naranja
#     else:
#         return "#FF4B4B"  # Rojo

# # 📌 Generar Predicciones y Mostrar Top 3 Sets
# if st.button("Generar Predicciones"):
#     if "PredictedInvestmentScore" not in df_filtrado.columns:
#         features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit',
#                     'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity',
#                     'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

#         # 📌 Asegurar que hay datos antes de predecir
#         if df_filtrado.shape[0] == 0:
#             st.error("❌ No hay sets disponibles para predecir. Prueba ajustando los filtros.")
#             st.stop()

#         df_filtrado.loc[:, "PredictedInvestmentScore"] = modelo.predict(df_filtrado[features])
#         df_filtrado = df_filtrado[df_filtrado["PredictedInvestmentScore"] > 0]

#         if df_filtrado.shape[0] < 3:
#             st.warning("⚠️ Menos de 3 sets cumplen con los criterios seleccionados. Mostrando los disponibles.")

#         df_filtrado = df_filtrado.sort_values(by="PredictedInvestmentScore", ascending=False).head(3)

    # st.subheader("📊 Top 3 Sets Más Rentables")
    # if not df_filtrado.empty:
    #     cols = st.columns(len(df_filtrado))
    #     for col, (_, row) in zip(cols, df_filtrado.iterrows()):
    #         with col:
    #             color = get_color(row["PredictedInvestmentScore"])
    #             st.markdown(f"""
    #                 <div style='background-color:{color}; padding:10px; border-radius:5px; text-align:center; margin-bottom:10px;'>
    #                     <strong>{row['SetName']}</strong>
    #                 </div>
    #             """, unsafe_allow_html=True)
    #             st.markdown(f"""
    #                 <div style='display: flex; justify-content: center;'>
    #                     <img src='{get_lego_image(row["Number"])}' width='100%'>
    #                 </div>
    #             """, unsafe_allow_html=True)
    #             st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
    #             st.write(f"**Tema:** {row['Theme']}")
    #             st.write(f"💰 **Precio:** ${row['USRetailPrice']:.2f}")
    #             url_lego = f"https://www.lego.com/en-us/product/{row['Number']}"
    #             st.markdown(f'<a href="{url_lego}" target="_blank"><button style="background-color:#ff4b4b; border:none; padding:10px; border-radius:5px; cursor:pointer; font-size:14px;">🛒 Comprar en LEGO</button></a>', unsafe_allow_html=True)
    #             st.write("---")
