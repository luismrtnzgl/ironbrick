import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from itertools import combinations
from data_processing import load_and_clean_data
from alerts import send_telegram_alert

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(file_path):
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        print(f"⚠️ Archivo no encontrado: {file_path}")
        return None

model_xgb_2y = load_model(os.path.join(BASE_DIR, "models/xgb_2y.pkl"))
model_xgb_5y = load_model(os.path.join(BASE_DIR, "models/xgb_5y.pkl"))

def find_best_combinations(df, budget, top_n=3):
    """
    Función para encontrar las mejores combinaciones de sets que maximicen la rentabilidad dentro del presupuesto.
    """
    best_combinations = []
    df_sorted = df.sort_values(by=["Pred_2Y"], ascending=False)
    
    # Aquí definimos que queremos probar combinaciones de hasta 3 sets
    for r in range(1, min(4, len(df_sorted) + 1)):  
        for combo in combinations(df_sorted.index, r):
            combo_df = df_sorted.loc[list(combo)]
            total_cost = combo_df["CurrentValueNew"].sum()
            total_pred_2y = combo_df["Pred_2Y"].sum()
            total_pred_5y = combo_df["Pred_5Y"].sum()
            
            if total_cost <= budget:
                best_combinations.append((combo_df, total_cost, total_pred_2y, total_pred_5y))
    
    best_combinations = sorted(best_combinations, key=lambda x: x[2], reverse=True)[:top_n]
    return best_combinations

def main():
    st.title("Recomendación de Inversión en Sets de LEGO")
    
    # Cargamos y limpiamos el dataset descargado de la API
    df = load_and_clean_data("data/scraped_lego_data.csv")
    st.write("### Datos Procesados")
    st.dataframe(df.head())
    
    # Seleccionamos el presupuesto
    budget = st.number_input("Introduce tu presupuesto en USD", min_value=0, value=100)
    
    # Filtramos los sets dentro del presupuesto
    df_budget = df[df["CurrentValueNew"] <= budget]
    
    if df_budget.empty:
        st.write("No hay sets disponibles dentro de este presupuesto.")
        return
    
    # Realizamos las predicciones para todos los sets dentro del presupuesto
    price_columns = [col for col in df.columns if col.startswith("Price_")]
    X_budget = df_budget[price_columns + ["RetailPriceUSD"]]
    df_budget["Pred_2Y"] = model_xgb_2y.predict(X_budget)
    df_budget["Pred_5Y"] = model_xgb_5y.predict(X_budget)
    
    # Filtramos los sets con alta probabilidad de revalorización
    df_invest = df_budget[(df_budget["Pred_2Y"] > df_budget["CurrentValueNew"] * 1.3) | (df_budget["Pred_5Y"] > df_budget["CurrentValueNew"] * 1.5)]
    
    if df_invest.empty:
        st.write("Lo sentimos, no hay sets recomendados para inversión dentro del presupuesto que nos has facilitado.")
    else:
        st.write("### Mejores combinaciones de inversión")
        best_combos = find_best_combinations(df_invest, budget)
        
        for i, (combo_df, total_cost, total_pred_2y, total_pred_5y) in enumerate(best_combos):
            st.write(f"#### Opción {i+1} (Costo Total: ${total_cost:.2f})")
            st.dataframe(combo_df[["SetName", "Year", "Theme", "CurrentValueNew", "Pred_2Y", "Pred_5Y"]])
            
            # Envío de alertas para cada set en la combinación
            for _, row in combo_df.iterrows():
                send_telegram_alert(row["SetName"], row["Pred_2Y"], row["Pred_5Y"])  
    
if __name__ == "__main__":
    main()
