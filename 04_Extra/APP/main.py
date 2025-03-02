import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import itertools

# 📌 Obtener la ruta absoluta del CSV
BASE_DIR = os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "04_Extra/APP/data/scraped_lego_data.csv")

# 📌 Verificar modelos
pkl_path_2y = os.path.join(BASE_DIR, "04_Extra/APP/models/xgb_2y.pkl")
pkl_path_5y = os.path.join(BASE_DIR, "04_Extra/APP/models/xgb_5y.pkl")

if not os.path.exists(pkl_path_2y) or not os.path.exists(pkl_path_5y):
    st.error("❌ No se encontraron los modelos .pkl en la carpeta 'models/'.")
    st.stop()

# 📌 Cargar modelos
@st.cache_resource
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

model_2y = load_model(pkl_path_2y)
model_5y = load_model(pkl_path_5y)
st.success("✅ Modelos cargados correctamente.")

# 📌 Procesar el CSV
@st.cache_data
def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["PriceDate"] = pd.to_datetime(df["PriceDate"], errors='coerce')
    df = df.dropna(subset=["PriceDate"])  # Eliminar solo fechas inválidas
    
    # Ordenar y asignar índices
    df_sorted = df.sort_values(by=['Number', 'PriceDate'])
    df_sorted['PriceIndex'] = df_sorted.groupby('Number').cumcount()
    
    # Pivotar precios
    df_transformed = df_sorted.pivot(index=['Number', 'SetName', 'Theme', 'Year', 'Pieces', 
                                            'RetailPriceUSD', 'CurrentValueNew', 'ForecastValueNew2Y', 
                                            'ForecastValueNew5Y'],
                                     columns='PriceIndex', values='PriceValue').reset_index()
    
    # Renombrar columnas de precios correctamente
    df_transformed.columns = [f'Price_{col+1}' if isinstance(col, int) else col for col in df_transformed.columns]
    
    # Seleccionar solo las primeras 12 columnas de precios
    price_columns = [f'Price_{i}' for i in range(1, 13)]
    df_transformed = df_transformed[['Number', 'SetName', 'Theme', 'Year', 'Pieces', 
                                     'RetailPriceUSD', 'CurrentValueNew', 'ForecastValueNew2Y', 
                                     'ForecastValueNew5Y'] + price_columns]
    
    # Eliminar filas con valores nulos
    df_transformed = df_transformed.dropna()
    
    return df_transformed

# 📌 Procesar el dataset
df_identification = process_csv(CSV_PATH)

# 📌 Generar predicciones
df_model = df_identification.drop(columns=['Number', 'SetName', 'Theme'], errors='ignore')

df_model = pd.get_dummies(df_model, drop_first=True)
expected_columns = model_2y.feature_names_in_
for col in expected_columns:
    if col not in df_model.columns:
        df_model[col] = 0

df_model = df_model[expected_columns]
df_identification['PredictedValue2Y'] = model_2y.predict(df_model)
df_identification['PredictedValue5Y'] = model_5y.predict(df_model)

st.success("✅ Predicciones generadas correctamente.")
st.dataframe(df_identification)
