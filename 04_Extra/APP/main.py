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
    st.write("🔍 Dataframe cargado (primeras filas):", df.head())
    st.write("📏 Dimensiones iniciales:", df.shape)
    
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
    
    st.write("🔄 Dataframe después del pivotado:", df_transformed.head())
    st.write("📏 Dimensiones después del pivotado:", df_transformed.shape)
    
    # Renombrar columnas de precios correctamente
    df_transformed.columns = [f'Price_{col+1}' if isinstance(col, int) else col for col in df_transformed.columns]
    
    # Seleccionar solo las primeras 12 columnas de precios
    price_columns = [f'Price_{i}' for i in range(1, 13)]
    df_transformed = df_transformed[['Number', 'SetName', 'Theme', 'Year', 'Pieces', 
                                     'RetailPriceUSD', 'CurrentValueNew', 'ForecastValueNew2Y', 
                                     'ForecastValueNew5Y'] + price_columns]
    
    # Reemplazar valores nulos en las columnas de precios por 0
    df_transformed[price_columns] = df_transformed[price_columns].fillna(0)
    df_transformed['Pieces'] = df_transformed['Pieces'].fillna(0)
    df_transformed['RetailPriceUSD'] = df_transformed['RetailPriceUSD'].fillna(0)
    df_transformed.loc[df_transformed['CurrentValueNew'] == 0, 'CurrentValueNew'] = df_transformed['RetailPriceUSD']
    
    st.write("📊 Dataframe antes de eliminar nulos:", df_transformed.shape)
    df_transformed = df_transformed.dropna()
    st.write("📊 Dataframe después de eliminar nulos:", df_transformed.shape)
    
    # Guardar un CSV temporal en Streamlit para verificar el dataframe final
    df_transformed.to_csv("df_debug.csv", index=False)
    st.write("📂 Se ha guardado un archivo CSV llamado 'df_debug.csv' para depuración.")
    
    return df_transformed

# 📌 Procesar el dataset
df_identification = process_csv(CSV_PATH)

st.write("✅ Dataframe final de identificación (primeras filas):", df_identification.head())
st.write("📏 Dimensiones finales:", df_identification.shape)

if df_identification.empty:
    st.error("❌ El dataframe de identificación está vacío después del procesamiento.")
else:
    st.success("✅ El dataframe tiene datos correctamente.")
    st.dataframe(df_identification)  # Mostrar en pantalla

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
