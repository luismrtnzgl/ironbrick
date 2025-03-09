import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
import pymongo #cambio erv

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

st.title("🎯 Recomendador de inversión en sets de LEGO retirados")
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
