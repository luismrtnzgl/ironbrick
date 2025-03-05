import telebot
import sqlite3
import pandas as pd
import joblib
import os
import requests
import time
import schedule

# 📌 Obtener el token del bot
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("❌ ERROR: No se encontró el TOKEN del bot de Telegram.")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# 📌 Cargar el modelo desde GitHub
modelo_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/05_Streamlit/models/stacking_model.pkl"
modelo_path = "/tmp/stacking_model.pkl"

if not os.path.exists(modelo_path):
    response = requests.get(modelo_url)
    with open(modelo_path, "wb") as f:
        f.write(response.content)

modelo = joblib.load(modelo_path)

# 📌 Cargar dataset y preprocesarlo
dataset_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/01_Data_Cleaning/df_lego_final_venta.csv"
df_lego = pd.read_csv(dataset_url)

def preprocess_data(df):
    df = df[df['USRetailPrice'] > 0].copy()

    exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
    df['Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)

    size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
    df['SizeCategory'] = df['SizeCategory'].map(size_category_mapping)

    df["PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]
    df["PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)
    df["YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

df_lego = preprocess_data(df_lego)

# 📌 Aplicar el modelo a todos los sets
features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 
            'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity', 
            'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

df_lego["PredictedInvestmentScore"] = modelo.predict(df_lego[features])

def enviar_recomendaciones():
    conn = sqlite3.connect("user_ironbrick.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT telegram_id FROM usuarios")
    usuarios = cursor.fetchall()
    conn.close()

    for user_id in usuarios:
        user_id = user_id[0]
        mensaje = "📊 *Ranking de inversión en LEGO*\n\n"
        for _, row in df_lego.sort_values(by="PredictedInvestmentScore", ascending=False).head(5).iterrows():
            mensaje += f"🧱 *{row['SetName']}* ({row['Number']})\n"
            mensaje += f"💰 *Precio:* ${row['USRetailPrice']}\n"
            mensaje += f"📈 *Rentabilidad:* {row['PredictedInvestmentScore']:.2f}\n\n"
        
        bot.send_message(user_id, mensaje, parse_mode="Markdown")

schedule.every(30).days.do(enviar_recomendaciones)

while True:
    schedule.run_pending()
    time.sleep(86400)  # Revisar cada 24 horas
