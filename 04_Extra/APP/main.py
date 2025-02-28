import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import itertools
import requests


# 📌 1. URL del dataset en GitHub (REEMPLAZA CON TU ENLACE RAW)
GITHUB_CSV_URL = "https://github.com/luismrtnzgl/ironbrick/blob/39a91c139ba3da906cd2ae6c6c9575c3161e72ab/04_Extra/APP/data/scraped_lego_data.csv"

# 📌 2. Verificar si los modelos existen antes de cargarlos
pkl_path_2y = "models/xgb_2y.pkl"
pkl_path_5y = "models/xgb_5y.pkl"

if not os.path.exists(pkl_path_2y) or not os.path.exists(pkl_path_5y):
    st.error("❌ No se encontraron los modelos .pkl en la carpeta 'models/'. Asegúrate de que los archivos existen.")
    st.stop()

# 📌 3. Función para cargar modelos pre-entrenados
@st.cache_resource
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# 📌 4. Cargar los modelos
model_2y = load_model(pkl_path_2y)
model_5y = load_model(pkl_path_5y)
st.success("✅ Modelos cargados correctamente.")

# 📌 5. Función para cargar y procesar el dataset desde GitHub
@st.cache_data
def load_and_process_github_csv(url):
    st.write("📥 Descargando datos desde GitHub...")
    
    # Descargar archivo desde GitHub
    df = pd.read_csv(url)

    # 📌 Guardar identificadores
    id_columns = ['Number', 'SetName', 'Theme', 'CurrentValueNew']
    df_identification = df[id_columns]

    # 📌 Mantener las columnas de precios históricos Price_1 a Price_12
    price_columns = [col for col in df.columns if col.startswith('Price_')]

    # 📌 Crear copia sin identificadores
    df_model = df.drop(columns=['Number', 'SetName', 'Theme'], errors='ignore')

    # 📌 Convertir variables categóricas en dummies (alinearlas con el modelo entrenado)
    df_model = pd.get_dummies(df_model, drop_first=True)

    # 📌 Asegurar que las columnas coincidan con las del modelo
    expected_columns = model_2y.feature_names_in_
    for col in expected_columns:
        if col not in df_model.columns:
            df_model[col] = 0  # Añadir columna faltante con ceros
    
    df_model = df_model[expected_columns]  # Ordenar columnas como el modelo espera

    return df_identification, df_model

# 📌 6. Función para encontrar combinaciones óptimas de inversión
def find_best_investments(df, budget, num_options=3):
    sets_list = df[['SetName', 'CurrentValueNew', 'PredictedValue2Y', 'PredictedValue5Y']].values.tolist()
    
    best_combinations = []
    
    # 📌 Generamos combinaciones de 1 hasta 4 sets
    for r in range(1, 5):
        for combination in itertools.combinations(sets_list, r):
            total_price = sum(item[1] for item in combination)
            total_return_2y = sum(item[2] for item in combination)
            total_return_5y = sum(item[3] for item in combination)

            if total_price <= budget:
                best_combinations.append((combination, total_return_2y, total_return_5y, total_price))

    # 📌 Ordenamos por mejor rentabilidad a 5 años
    best_combinations.sort(key=lambda x: x[2], reverse=True)

    return best_combinations[:num_options]  # Devolver las 3 mejores combinaciones

# 📌 7. Interfaz de Streamlit
st.title("💰 Recomendador de Inversiones en LEGO (desde GitHub)")

# 📌 Cargar y procesar el dataset desde GitHub
df_identification, df_model = load_and_process_github_csv(GITHUB_CSV_URL)
st.success("✅ Datos cargados correctamente")

# 📌 Hacer predicciones
df_identification['PredictedValue2Y'] = model_2y.predict(df_model)
df_identification['PredictedValue5Y'] = model_5y.predict(df_model)

# 📌 Mostrar el dataframe con predicciones
st.subheader("📊 Sets de LEGO con Predicción de Revalorización")
st.dataframe(df_identification)

# 📌 Selección de presupuesto
budget = st.number_input("Introduce tu presupuesto ($)", min_value=10, value=200)

if st.button("🔍 Buscar inversiones óptimas"):
    best_options = find_best_investments(df_identification, budget)

    if not best_options:
        st.warning("⚠️ No se encontraron combinaciones dentro de tu presupuesto.")
    else:
        st.subheader("💡 Mejores opciones de inversión")
        for i, (combo, ret_2y, ret_5y, price) in enumerate(best_options, 1):
            st.write(f"**Opción {i}:**")
            st.write(f"💵 **Precio Total:** ${price:.2f}")
            st.write(f"📈 **Valor estimado en 2 años:** ${ret_2y:.2f}")
            st.write(f"🚀 **Valor estimado en 5 años:** ${ret_5y:.2f}")
            st.write("🧩 **Sets incluidos:**")
            for set_name, price, _, _ in combo:
                st.write(f"- {set_name} (${price:.2f})")
            st.write("---")
