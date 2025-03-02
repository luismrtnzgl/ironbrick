import os
import streamlit as st
import pandas as pd

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
    st.dataframe(df.head(20))  # Mostrar las primeras 20 filas
except Exception as e:
    st.error(f"❌ ERROR al leer el archivo CSV: {e}")
    st.stop()

# 📌 Paso 2: Procesamiento del dataset
st.write("🔄 Iniciando procesamiento de datos...")
try:
    # 1️⃣ Convertir PriceDate a datetime
    df["PriceDate"] = pd.to_datetime(df["PriceDate"], errors='coerce')
    df = df.dropna(subset=["PriceDate"])
    st.write("📆 PriceDate convertido a datetime.")
    
    # 2️⃣ Ordenar por 'Number' y 'PriceDate'
    df_sorted = df.sort_values(by=['Number', 'PriceDate'])
    
    # 3️⃣ Crear índice secuencial de precios
    df_sorted['PriceIndex'] = df_sorted.groupby('Number').cumcount()
    
    # 4️⃣ Pivotar la tabla para estructurar los precios correctamente
    df_transformed = df_sorted.pivot(index=['Number', 'SetName', 'Theme', 'Year', 'Pieces', 
                                            'RetailPriceUSD', 'CurrentValueNew', 'ForecastValueNew2Y', 
                                            'ForecastValueNew5Y'],
                                     columns='PriceIndex', values='PriceValue').reset_index()
    st.write("📊 Datos pivotados correctamente.")
    
    # 5️⃣ Renombrar columnas de precios
    df_transformed.columns = [f'Price_{col+1}' if isinstance(col, int) else col for col in df_transformed.columns]
    
    # 6️⃣ Mantener solo las primeras 12 columnas de precios
    price_columns = [f'Price_{i}' for i in range(1, 13)]
    df_transformed = df_transformed[['Number', 'SetName', 'Theme', 'Year', 'Pieces', 
                                     'RetailPriceUSD', 'CurrentValueNew', 'ForecastValueNew2Y', 
                                     'ForecastValueNew5Y'] + price_columns]
    
    # 7️⃣ Llenar valores nulos en las columnas de precios con 0
    df_transformed[price_columns] = df_transformed[price_columns].fillna(0)
    df_transformed['Pieces'] = df_transformed['Pieces'].fillna(0)
    df_transformed['RetailPriceUSD'] = df_transformed['RetailPriceUSD'].fillna(0)
    df_transformed.loc[df_transformed['CurrentValueNew'] == 0, 'CurrentValueNew'] = df_transformed['RetailPriceUSD']
    
    # 8️⃣ Eliminar filas con valores nulos restantes
    df_transformed = df_transformed.dropna()
    
    st.write("✅ Procesamiento de datos completado.")
    st.write("📏 Dimensiones después del procesamiento:", df_transformed.shape)
    st.dataframe(df_transformed.head(20))
    
except Exception as e:
    st.error(f"❌ ERROR durante el procesamiento de datos: {e}")
    st.stop()
