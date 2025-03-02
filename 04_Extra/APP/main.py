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
df_transformed.loc[:, 'Pieces'] = df_transformed['Pieces'].fillna(0)
df_transformed.loc[:, 'RetailPriceUSD'] = df_transformed['RetailPriceUSD'].fillna(0)
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

# 📌 Optimización: Seleccionar los 10 mejores sets primero
df_top_sets = df_filtrado.sort_values(by="PredictedValue5Y", ascending=False).head(10)

# 📌 Generar combinaciones de inversión más eficientes
def encontrar_mejores_inversiones(df, presupuesto, num_opciones=3):
    sets_lista = df[['SetName', 'CurrentValueNew', 'PredictedValue2Y', 'PredictedValue5Y']].values.tolist()
    mejores_combinaciones = []
    
    for r in range(1, 4):  # Limitar a combinaciones de 1 a 3 sets para optimizar tiempo
        for combinacion in itertools.combinations(sets_lista, r):
            total_precio = sum(item[1] for item in combinacion)
            retorno_2y = sum(item[2] for item in combinacion)
            retorno_5y = sum(item[3] for item in combinacion)
            
            if total_precio <= presupuesto:
                mejores_combinaciones.append((combinacion, retorno_2y, retorno_5y, total_precio))
    
    mejores_combinaciones.sort(key=lambda x: x[2], reverse=True)
    return mejores_combinaciones[:num_opciones]

# 📌 Mostrar las mejores opciones de inversión
if st.sidebar.button("🔍 Buscar inversiones óptimas"):
    opciones = encontrar_mejores_inversiones(df_top_sets, presupuesto)
    
    if not opciones:
        st.warning("⚠️ No se encontraron combinaciones dentro de tu presupuesto.")
    else:
        st.subheader("💡 Mejores opciones de inversión")
        for i, (combo, ret_2y, ret_5y, precio) in enumerate(opciones, 1):
            st.write(f"**Opción {i}:**")
            st.write(f"💵 **Precio Total:** ${precio:.2f}")
            st.write(f"📈 **Valor estimado en 2 años:** ${ret_2y:.2f}")
            st.write(f"🚀 **Valor estimado en 5 años:** ${ret_5y:.2f}")
            st.write("🧩 **Sets incluidos:**")
            for set_name, price, _, _ in combo:
                st.write(f"- {set_name} (${price:.2f})")
            st.write("---")
