import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# 🔹 Definir preprocess_data antes de cargar joblib
def preprocess_data(df):
    """Función de preprocesamiento aplicada antes de entrenar el modelo"""
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
    df.infer_objects(copy=False)  
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

# Asegurar rutas absolutas para Streamlit Cloud
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models/stacking_model_compressed.pkl")
PREPROCESS_PATH = os.path.join(BASE_DIR, "models/preprocessing.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data/df_lego_final_venta.csv")

# 🔹 Cargar el modelo y la función de preprocesamiento con joblib
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_preprocessing():
    """Carga la función de preprocesamiento asegurando que preprocess_data está disponible"""
    return joblib.load(PREPROCESS_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    preprocess_function = load_preprocessing()
    return preprocess_function(df)  # Aplicar la función de preprocesamiento

# Cargar recursos
model = load_model()
df_ranking = load_data()
