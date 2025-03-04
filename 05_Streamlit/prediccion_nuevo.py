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
- 🟢 **Verde**: Baja probabilidad de pérdida, inversión segura.
- 🟡 **Amarillo**: Riesgo medio, potencial de revalorización con incertidumbre.
- 🔴 **Rojo**: Riesgo alto, posibilidad de baja rentabilidad.
""")

st.subheader("Configura tu Inversión en LEGO")
presupuesto_min, presupuesto_max = st.slider("Selecciona el rango de presupuesto (USD)", 
                                             min_value=100, max_value=1000, value=(500, 800), step=10)

themes_options = ["Todos"] + sorted(df_ranking["Theme"].unique().tolist())
selected_themes = st.multiselect("Selecciona los Themes de Interés", themes_options, default=["Todos"])

df_filtrado = df_ranking if "Todos" in selected_themes else df_ranking[df_ranking["Theme"].isin(selected_themes)]

df_filtrado = df_filtrado[(df_filtrado["USRetailPrice"] >= presupuesto_min) & 
                          (df_filtrado["USRetailPrice"] <= presupuesto_max)]

def get_lego_image(set_number):
    return f"https://images.brickset.com/sets/images/{set_number}-1.jpg"

def get_color(score):
    if score > 15:
        return "#28B463"  # Verde
    elif score > 6:
        return "#FFC300"  # Amarillo
    else:
        return "#FF5733"  # Naranja

if st.button("Generar Predicciones"):
    if "PredictedInvestmentScore" not in df_filtrado.columns:
        st.warning("⚠️ No se encontró 'PredictedInvestmentScore', aplicando modelo...")
        model = load_model()
        features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 'ResaleDemand', 
                    'AnnualPriceIncrease', 'Exclusivity', 'SizeCategory', 'PricePerPiece', 
                    'PricePerMinifig', 'YearsOnMarket', 'InteractionFeature']
        df_filtrado.loc[:, "PredictedInvestmentScore"] = model.predict(df_filtrado[features].values)
        st.success("✅ PredictedInvestmentScore generado correctamente.")
    
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
                <div style='background-color:{color}; padding:10px; border-radius:5px; text-align:center;'>
                    <strong>{row['SetName']}</strong>
                </div>
            """, unsafe_allow_html=True)
            st.image(get_lego_image(row["Number"]), width=150)
            st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)
st.write(f"💰 **Precio:** ${row['USRetailPrice']:.2f}")
st.write(f"📊 **Predicted Investment Score:** {row['PredictedInvestmentScore']:.2f}")
url_lego = f"https://www.lego.com/en-us/product/{row['Number']}"
st.button("🛒 Comprar en LEGO", key=f"buy_{row['Number']}", on_click=lambda: st.markdown(f'<meta http-equiv="refresh" content="0; url={url_lego}" />', unsafe_allow_html=True))
st.write("---")
