import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import itertools

# 📌 Obtener la ruta del archivo CSV
BASE_DIR = os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "04_Extra/APP/data/scraped_lego_data.csv")

# 📌 Verificar si el archivo existe
st.write("📂 Ruta del archivo CSV:", CSV_PATH)
if not os.path.exists(CSV_PATH):
    st.error("❌ ERROR: El archivo CSV NO EXISTE en la ruta especificada.")
    st.stop()

# 📌 Cargar el archivo CSV sin modificaciones
try:
    df = pd.read_csv(CSV_PATH)
    st.success("✅ Archivo CSV cargado correctamente.")
    st.write("📏 Dimensiones del archivo:", df.shape)
except Exception as e:
    st.error(f"❌ ERROR al leer el archivo CSV: {e}")
    st.stop()

# 📌 Paso 2: Procesamiento del dataset
st.write("🔄 Iniciando procesamiento de datos...")
try:
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
    st.write("✅ Procesamiento de datos completado.")
    st.write("📏 Dimensiones después del procesamiento:", df_transformed.shape)
except Exception as e:
    st.error(f"❌ ERROR durante el procesamiento de datos: {e}")
    st.stop()

# 📌 Paso 3: Cargar modelos y generar predicciones
st.write("🔄 Cargando modelos de predicción...")
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

# 📌 Preparar dataset para predicciones
st.write("🔄 Preparando dataset para predicción...")
df_identification = df_transformed[['Number', 'SetName', 'Theme', 'CurrentValueNew']]
df_model = df_transformed.drop(columns=['Number', 'SetName', 'Theme'], errors='ignore')
df_model = pd.get_dummies(df_model, drop_first=True)

# 📌 Asegurar que el dataset tenga las mismas columnas que el modelo
expected_columns = model_2y.feature_names_in_
for col in expected_columns:
    if col not in df_model.columns:
        df_model[col] = 0

df_model = df_model[expected_columns]

# 📌 Generar predicciones
df_identification['PredictedValue2Y'] = model_2y.predict(df_model)
df_identification['PredictedValue5Y'] = model_5y.predict(df_model)
st.success("✅ Predicciones generadas correctamente.")
st.write("📏 Dimensiones del dataframe con predicciones:", df_identification.shape)
st.dataframe(df_identification.head(20))
