import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import itertools
import requests

# 📌 Obtener la ruta absoluta del CSV en la carpeta 'data/'
BASE_DIR = os.getcwd()
CSV_PATH = os.path.join(BASE_DIR, "04_Extra/APP/data/scraped_lego_data.csv")

# 📌 2. Verificar si los modelos existen antes de cargarlos
pkl_path_2y = os.path.join(BASE_DIR, "04_Extra/APP/models/xgb_2y.pkl")
pkl_path_5y = os.path.join(BASE_DIR, "04_Extra/APP/models/xgb_5y.pkl")

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

# 📌 5. Función para cargar y procesar el dataset
@st.cache_data
def process_csv(csv_path):
    df_lego_scrap_brickeco = pd.read_csv(csv_path)

    # 📌 1. Ordenamos por 'Number' y 'PriceDate' para garantizar el orden cronológico
    df_sorted = df_lego_scrap_brickeco.sort_values(by=['Number', 'PriceDate'], ascending=[True, True])

    # 📌 2. Asignamos un índice secuencial correcto para cada set sin saltos en los precios
    df_sorted['PriceIndex'] = df_sorted.groupby('Number').cumcount()

    # 📌 3. Hacemos un pivotable para reorganizar los precios correctamente
    df_transformed = df_sorted.pivot(index=['Number', 'SetName', 'Theme', 'Year', 'Pieces', 
                                            'RetailPriceUSD', 'CurrentValueNew', 'ForecastValueNew2Y', 
                                            'ForecastValueNew5Y'],
                                     columns='PriceIndex', values='PriceValue').reset_index()

    # 📌 4. Renombramos las columnas de precios correctamente
    df_transformed.columns = [f'Price_{col+1}' if isinstance(col, int) else col for col in df_transformed.columns]

    # 📌 5. Identificamos las columnas de precios
    price_columns = [col for col in df_transformed.columns if col.startswith("Price_")]

    # 📌 6. Reordenamos los precios dentro de cada fila para eliminar espacios vacíos
    def reorder_prices(row):
        prices = row[price_columns].dropna().values
        new_row = [None] * len(price_columns)
        new_row[:len(prices)] = prices
        return pd.Series(new_row, index=price_columns)

    df_transformed[price_columns] = df_transformed.apply(reorder_prices, axis=1)

    # 📌 7. Eliminamos las columnas innecesarias
    columns_to_drop = ['Minifigs', 'RollingGrowthLastYear', 'RollingGrowth12M', 'CurrentValueUsed', 'Currency', 'URL', 'PriceType']
    price_columns_to_drop = [col for col in df_transformed.columns if col.startswith('Price_') and 13 <= int(col.split('_')[1]) <= 280]
    columns_to_drop.extend(price_columns_to_drop)

    df_transformed_limpia = df_transformed.drop(columns=[col for col in columns_to_drop if col in df_transformed.columns])

    columns_to_fill = [f'Price_{i}' for i in range(1, 13)]
    df_transformed_limpia[columns_to_fill] = df_transformed_limpia[columns_to_fill].fillna(0)

    df_transformed_limpia['Pieces'] = df_transformed_limpia['Pieces'].fillna(0)

    df_transformed_limpia['RetailPriceUSD'] = df_transformed_limpia['RetailPriceUSD'].fillna(0)

    df_transformed_limpia.loc[df_transformed_limpia['CurrentValueNew'] == 0, 'CurrentValueNew'] = df_transformed_limpia['RetailPriceUSD']

    # 📌 8. Eliminamos filas con valores nulos
    #df_transformed_limpia = df_transformed_limpia.dropna()

    # 📌 9. Guardar identificadores para después
    id_columns = ['Number', 'SetName', 'Theme', 'CurrentValueNew']
    df_identification = df_transformed_limpia[id_columns]

    # 📌 10. Preparamos `df_model` para el modelo de predicción
    df_model = df_transformed_limpia.drop(columns=['Number', 'SetName', 'Theme'], errors='ignore')
    df_model = pd.get_dummies(df_model, drop_first=True)  # Convertir variables categóricas

    # 📌 11. Asegurar que `df_model` tenga las mismas columnas que el modelo
    expected_columns = model_2y.feature_names_in_
    for col in expected_columns:
        if col not in df_model.columns:
            df_model[col] = 0
    df_model = df_model[expected_columns]  # Ordenar las columnas correctamente

    return df_identification, df_model

# 📌 Procesar el CSV antes de predecir
df_identification, df_model = process_csv(CSV_PATH)

# 📌 Generar predicciones
df_identification['PredictedValue2Y'] = model_2y.predict(df_model)
df_identification['PredictedValue5Y'] = model_5y.predict(df_model)

st.success("✅ Predicciones generadas correctamente.")

# 📌 Mostrar resultados en Streamlit
st.subheader("📊 Predicción de revalorización de Sets LEGO")
st.dataframe(df_identification)

# 📌 Selección de presupuesto
budget = st.number_input("Introduce tu presupuesto ($)", min_value=10, value=200)

# 📌 Función para encontrar combinaciones óptimas de inversión
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