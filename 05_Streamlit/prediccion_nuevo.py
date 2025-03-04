import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Función de preprocesamiento
def preprocess_data(df):
    df = df[df['USRetailPrice'] > 0].copy()
    
    exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
    df['Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)
    
    size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
    df['SizeCategory'] = df['SizeCategory'].map(size_category_mapping)
    
    df["PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]
    df["PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)
    df["YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]
    df["InteractionFeature"] = df["PricePerPiece"] * df["YearsOnMarket"]
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models/stacking_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data/df_lego_final_venta.csv")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)
    return df

df_ranking = load_data()

st.title("Plataforma de Recomendación de Inversión en LEGO 📊")

st.markdown("""
### Código de Color para Evaluación de Riesgo:
- 🟢 **Verde**: Alta probabilidad de revalorización.
- 🟡 **Amarillo**: Potencial de revalorización con un riesgo medio.
- 🟠 **Naranja**: Posibilidad de baja rentabilidad.
""")

st.subheader("Configura tu Inversión en LEGO")
presupuesto_min, presupuesto_max = st.slider("Selecciona el rango de presupuesto (USD)", 
                                             min_value=10, max_value=1500, value=(10, 200), step=10)

themes_options = ["Todos"] + sorted(df_ranking["Theme"].unique().tolist())
selected_themes = st.multiselect("Selecciona los Themes de Interés", themes_options, default=["Todos"])

df_filtrado = df_ranking if "Todos" in selected_themes else df_ranking[df_ranking["Theme"].isin(selected_themes)]

df_filtrado = df_filtrado[(df_filtrado["USRetailPrice"] >= presupuesto_min) & 
                          (df_filtrado["USRetailPrice"] <= presupuesto_max)]

def get_lego_image(set_number):
    return f"https://images.brickset.com/sets/images/{set_number}-1.jpg"

def get_color(score):
    if score > 10:
        return "#00736d"  # Verde
    elif score > 3:
        return "#FFC300"  # Amarillo
    else:
        return "#FF9944"  # Naranja

if st.button("Generar Predicciones"):
    if "PredictedInvestmentScore" not in df_filtrado.columns:
        model = load_model()
        features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 'ResaleDemand', 
                    'AnnualPriceIncrease', 'Exclusivity', 'SizeCategory', 'PricePerPiece', 
                    'PricePerMinifig', 'YearsOnMarket', 'InteractionFeature']
        df_filtrado.loc[:, "PredictedInvestmentScore"] = model.predict(df_filtrado[features].values)
    
    df_filtrado = df_filtrado[df_filtrado["PredictedInvestmentScore"] > 1]
    if df_filtrado.shape[0] < 3:
        st.warning("⚠️ Menos de 3 sets cumplen con los criterios seleccionados. Mostrando los disponibles.")
    df_filtrado = df_filtrado.sort_values(by="PredictedInvestmentScore", ascending=False).head(3)
    
    st.subheader("Top 3 Sets Más Rentables")
    cols = st.columns(len(df_filtrado))
    for col, (_, row) in zip(cols, df_filtrado.iterrows()):
        with col:
            color = get_color(row["PredictedInvestmentScore"])
            st.markdown(f"""
                <div style='background-color:{color}; padding:10px; border-radius:5px; text-align:center; margin-bottom:10px;'>
                    <strong>{row['SetName']}</strong>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style='display: flex; justify-content: center;'>
                    <img src='{get_lego_image(row["Number"])}' width='100%'>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
            st.write(f"**Tema:** {row['Theme']}")
            st.write(f"💰 **Precio:** ${row['USRetailPrice']:.2f}")
            url_lego = f"https://www.lego.com/en-us/product/{row['Number']}"
            st.markdown(f'<a href="{url_lego}" target="_blank"><button style="background-color:#ff4b4b; border:none; padding:10px; border-radius:5px; cursor:pointer; font-size:14px;">🛒 Comprar en LEGO</button></a>', unsafe_allow_html=True)
            st.write("---")
