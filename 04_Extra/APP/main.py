import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import itertools
import plotly.express as px

# 📌 Obtener la ruta del archivo CSV
BASE_DIR = os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "04_Extra/APP/data/scraped_lego_data.csv")

# 📌 Verificar si el archivo existe
st.write("📂 Ruta del archivo CSV:", CSV_PATH)
if not os.path.exists(CSV_PATH):
    st.error("❌ ERROR: El archivo CSV NO EXISTE en la ruta especificada.")
    st.stop()

# 📌 Cargar el archivo CSV
try:
    df = pd.read_csv(CSV_PATH)
    st.success("✅ Archivo CSV cargado correctamente.")
except Exception as e:
    st.error(f"❌ ERROR al leer el archivo CSV: {e}")
    st.stop()

# 📌 Procesar el dataset
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
df_transformed = df_transformed[['Number', 'SetName', 'Theme', 'Year', 'Pieces', 
                                 'RetailPriceUSD', 'CurrentValueNew', 'ForecastValueNew2Y', 
                                 'ForecastValueNew5Y'] + price_columns]
df_transformed[price_columns] = df_transformed[price_columns].fillna(0)
df_transformed['Pieces'] = df_transformed['Pieces'].fillna(0)
df_transformed['RetailPriceUSD'] = df_transformed['RetailPriceUSD'].fillna(0)
df_transformed.loc[df_transformed['CurrentValueNew'] == 0, 'CurrentValueNew'] = df_transformed['RetailPriceUSD']
df_transformed = df_transformed.dropna()

# 📌 Cargar modelos de predicción
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
st.success("✅ Modelos cargados correctamente.")

# 📌 Generar predicciones
df_identification = df_transformed[['Number', 'SetName', 'Theme', 'CurrentValueNew']]
df_model = df_transformed.drop(columns=['Number', 'SetName', 'Theme'], errors='ignore')
df_model = pd.get_dummies(df_model, drop_first=True)
expected_columns = model_2y.feature_names_in_
for col in expected_columns:
    if col not in df_model.columns:
        df_model[col] = 0
df_model = df_model[expected_columns]
df_identification['PredictedValue2Y'] = model_2y.predict(df_model)
df_identification['PredictedValue5Y'] = model_5y.predict(df_model)
st.success("✅ Predicciones generadas correctamente.")

# 📌 Interfaz de usuario en Streamlit
st.title("🔍 Análisis de Revalorización de Sets LEGO")
st.sidebar.header("Filtros de búsqueda")

# 📌 Filtro por tema
tema_seleccionado = st.sidebar.selectbox("Selecciona un tema", ["Todos"] + sorted(df_identification["Theme"].unique()))

# 📌 Filtro por presupuesto
presupuesto = st.sidebar.slider("Presupuesto máximo ($)", min_value=10, max_value=1000, value=200)

# 📌 Filtrar el dataframe
df_filtrado = df_identification.copy()
if tema_seleccionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Theme"] == tema_seleccionado]
df_filtrado = df_filtrado[df_filtrado["CurrentValueNew"] <= presupuesto]
st.write("📊 Resultados filtrados:", df_filtrado.shape)
st.dataframe(df_filtrado)

# 📌 Gráfico de evolución de precios
st.subheader("📈 Evolución de precios")
fig = px.line(df_transformed.melt(id_vars=['Number', 'SetName'], value_vars=price_columns, var_name='Tiempo', value_name='Precio'),
              x='Tiempo', y='Precio', color='SetName', title="Evolución de precios en el tiempo")
st.plotly_chart(fig)
