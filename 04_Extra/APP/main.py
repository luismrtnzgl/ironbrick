import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt

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

# 📌 Seleccionar las columnas de precios
price_columns = [f'Price_{i}' for i in range(1, 13)]
dias = [-15 * i for i in range(1, 13)]  # -15, -30, -45, ..., -180
renamed_columns = {f'Price_{i}': f"{dias[i-1]} días" for i in range(1, 13)}

# 📌 Mantener el dataframe sin cambios pero usar nombres alternativos en el gráfico
df_transformed.fillna(0, inplace=True)

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

# 📌 Selección de temas
st.subheader("🎯 Selecciona tus temas de interés")
temas_disponibles = sorted(df_identification["Theme"].unique())
temas_seleccionados = st.multiselect("Selecciona los temas de interés", ["Todos"] + temas_disponibles, default=["Todos"])

if "Todos" in temas_seleccionados:
    temas_seleccionados = temas_disponibles  

# 📌 Filtro por presupuesto
presupuesto = st.slider("Presupuesto máximo ($)", min_value=100, max_value=2000, value=200, step=10)

# 📌 Filtrar el dataframe
df_filtrado = df_transformed[df_transformed["Theme"].isin(temas_seleccionados)]
df_filtrado = df_filtrado[df_filtrado["CurrentValueNew"] <= presupuesto]

# 📌 Mostrar gráfico de evolución de precios solo con nombres alternativos en el gráfico
if not df_filtrado.empty:
    st.subheader("📈 Evolución de precios de sets seleccionados")

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, row in df_filtrado.iterrows():
        precios = row[price_columns].values
        ax.plot(dias, precios, marker='o', label=row["SetName"])

    ax.set_xlabel("Días en el pasado")
    ax.set_ylabel("Precio ($)")
    ax.set_title("Evolución de precios de los sets seleccionados")
    ax.legend(loc="upper left", fontsize="small", bbox_to_anchor=(1, 1))
    ax.grid(True)
    st.pyplot(fig)

# 📌 Optimización: Seleccionar los 10 mejores sets primero
df_top_sets = df_filtrado.sort_values(by="ForecastValueNew5Y", ascending=False).head(10)

# 📌 Buscar combinaciones óptimas de inversión
def encontrar_mejores_inversiones(df, presupuesto, num_opciones=3):
    sets_lista = df[['SetName', 'CurrentValueNew', 'ForecastValueNew2Y', 'ForecastValueNew5Y']].values.tolist()
    mejores_combinaciones = []
    
    for r in range(1, 5):  
        for combinacion in itertools.combinations(sets_lista, r):
            total_precio = sum(item[1] for item in combinacion)
            retorno_5y = sum(item[3] for item in combinacion)
            
            if total_precio <= presupuesto:
                mejores_combinaciones.append((combinacion, retorno_5y, total_precio))
    
    mejores_combinaciones.sort(key=lambda x: x[1], reverse=True)
    return mejores_combinaciones[:num_opciones]

# 📌 Mostrar inversiones óptimas
if st.button("🔍 Buscar inversiones óptimas"):
    opciones = encontrar_mejores_inversiones(df_top_sets, presupuesto)
    
    if not opciones:
        st.warning("⚠️ No se encontraron combinaciones dentro de tu presupuesto.")
    else:
        st.subheader("💡 Mejores opciones de inversión")
        for i, (combo, ret_5y, precio) in enumerate(opciones, 1):
            st.write(f"**Opción {i}:** 💵 Inversión: ${precio:.2f} 🚀 Rentabilidad 5Y: {ret_5y:.2f}%")
