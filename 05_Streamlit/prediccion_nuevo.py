import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 🔹 Función de preprocesamiento (asegura que las columnas necesarias estén presentes)
def preprocess_data(df):
    df = df[df['USRetailPrice'] > 0].copy()

    exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
    df.loc[:, 'Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)

    size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
    df.loc[:, 'SizeCategory'] = df['SizeCategory'].map(size_category_mapping)

    df.loc[:, "PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]
    df.loc[:, "PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)
    df.loc[:, "YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]
    df.loc[:, "InteractionFeature"] = df["PricePerPiece"] * df["YearsOnMarket"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

# 🔹 Definir rutas absolutas
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models/stacking_model_compressed.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data/df_lego_final_venta.csv")

# 🔹 Cargar el modelo con joblib
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# 🔹 Cargar y procesar los datos
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return preprocess_data(df)  # Aplicar preprocesamiento

# 🔹 Cargar recursos
df_ranking = load_data()

# 🔹 Streamlit App
st.title("Plataforma de Recomendación de Inversión en LEGO 📊")

# 🔹 Selección de presupuesto con rango (Máximo 1000, pasos de 10)
st.subheader("Configura tu Inversión en LEGO")
presupuesto_min, presupuesto_max = st.slider("Selecciona el rango de presupuesto (USD)", 
                                             min_value=100, max_value=1000, value=(200, 800), step=10)

# 🔹 Selección de temas con opción "Todos"
themes_options = ["Todos"] + sorted(df_ranking["Theme"].unique().tolist())
selected_themes = st.multiselect("Selecciona los Themes de Interés", themes_options, default=["Todos"])

# 🔹 Botón para generar la predicción
if st.button("Generar Predicción"):
    
    # 🔹 Filtrar sets según selección de temas
    if "Todos" in selected_themes:
        df_filtrado = df_ranking
    else:
        df_filtrado = df_ranking[df_ranking["Theme"].isin(selected_themes)].copy()

    # 🔹 Filtrar por presupuesto
    df_filtrado = df_filtrado[(df_filtrado["USRetailPrice"] >= presupuesto_min) & 
                              (df_filtrado["USRetailPrice"] <= presupuesto_max)]

    # 🔹 Generar PredictedInvestmentScore si no está presente
    if "PredictedInvestmentScore" not in df_filtrado.columns:
        st.warning("⚠️ No se encontró 'PredictedInvestmentScore', aplicando modelo...")
        model = load_model()
        features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 'ResaleDemand', 
                    'AnnualPriceIncrease', 'Exclusivity', 'SizeCategory', 'PricePerPiece', 
                    'PricePerMinifig', 'YearsOnMarket', 'InteractionFeature']
        df_filtrado["PredictedInvestmentScore"] = model.predict(df_filtrado[features])
        st.success("✅ PredictedInvestmentScore generado correctamente.")

    # 🔹 Filtrar sets con un `PredictedInvestmentScore` mayor a 1
    df_filtrado = df_filtrado[df_filtrado["PredictedInvestmentScore"] > 1]

    # 🔹 Generar combinaciones de inversión
    st.subheader("Opciones de Inversión")

    def generar_opciones_inversion(df, n_opciones=3):
        opciones = []
        df_sorted = df.sort_values(by="PredictedInvestmentScore", ascending=False)  # Priorizar mejores sets

        for _ in range(n_opciones):
            inversion = []
            for _, row in df_sorted.iterrows():
                if len(inversion) < 3:  # Máximo 3 sets por opción
                    inversion.append(row)
            opciones.append(pd.DataFrame(inversion))

        return opciones

    # 🔹 Generar 3 opciones de inversión
    opciones = generar_opciones_inversion(df_filtrado, n_opciones=3)

    # 🔹 Función para obtener la imagen del set desde Brickset
    def get_brickset_image(set_number):
        return f"https://images.brickset.com/sets/images/{set_number}-1.jpg"

    # 🔹 Definir colores para la seguridad de la inversión
    def get_color(score):
        if score > 15:
            return "#28B463"  # Verde
        elif score > 6:
            return "#FFC300"  # Amarillo
        else:
            return "#FF5733"  # Naranja

    # 🔹 Mostrar opciones de inversión en columnas
    for i, df_opcion in enumerate(opciones):
        if not df_opcion.empty:
            num_sets = len(df_opcion)  # Cantidad de sets en la opción
            cols = st.columns(num_sets)  # Crear exactamente el número de columnas necesarias

            for idx, row in df_opcion.iterrows():
                if idx < num_sets:  # Evita IndexError asegurando que idx no supere cols
                    with cols[idx]:
                        set_number = row["Number"]
                        image_url = f"https://images.brickset.com/sets/images/{set_number}-1.jpg"
                        lego_url = row["LEGO_URL"]  # Enlace a la tienda de LEGO

                        st.image(image_url, width=150)
                        st.markdown(f"### {row['SetName']}")
                        st.write(f"💰 **Precio:** ${row['USRetailPrice']:.2f}")
                        st.markdown(f"""
                            <div style='background-color:{get_color(row["PredictedInvestmentScore"])}; padding:10px; border-radius:5px; text-align:center;'>
                                <strong>Investment Score: {row["PredictedInvestmentScore"]:.2f}</strong>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"[🛒 Comprar en LEGO]({lego_url})", unsafe_allow_html=True)