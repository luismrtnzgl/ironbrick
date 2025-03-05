import telebot
import sqlite3
import pandas as pd
import joblib
import os
import time
import schedule

# 📌 Obtener el token de Telegram desde las variables de entorno
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("❌ ERROR: No se encontró el TOKEN del bot de Telegram.")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# 📌 Cargar el modelo desde GitHub
modelo_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/05_Streamlit/models/stacking_model.pkl"
modelo_path = "/tmp/stacking_model.pkl"

if not os.path.exists(modelo_path):
    import requests
    response = requests.get(modelo_url)
    with open(modelo_path, "wb") as f:
        f.write(response.content)

modelo = joblib.load(modelo_path)

# 📌 Cargar dataset de sets en venta desde GitHub
dataset_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/01_Data_Cleaning/df_lego_final_venta.csv"
df_lego = pd.read_csv(dataset_url)

# 📌 Aplicar el modelo para predecir oportunidades de inversión
features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 
            'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity', 
            'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

df_lego["PredictedInvestmentScore"] = modelo.predict(df_lego[features])

# 📌 Función para obtener recomendaciones personalizadas
def obtener_recomendaciones(telegram_id):
    """Filtra sets según los criterios del usuario en la base de datos."""
    conn = sqlite3.connect("user_ironbrick.db")  # Cambiado a la BD correcta
    cursor = conn.cursor()
    
    cursor.execute("SELECT presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios WHERE telegram_id = ?", (telegram_id,))
    usuario = cursor.fetchone()
    conn.close()

    if not usuario:
        return None

    presupuesto_min, presupuesto_max, temas_favoritos = usuario
    temas_lista = temas_favoritos.split(",")

    df_filtrado = df_lego[
        (df_lego["USRetailPrice"] >= presupuesto_min) & 
        (df_lego["USRetailPrice"] <= presupuesto_max)
    ]

    if "Todos" not in temas_lista:
        df_filtrado = df_filtrado[df_filtrado["Theme"].isin(temas_lista)]

    return df_filtrado.sort_values(by="PredictedInvestmentScore", ascending=False).head(5)

# 📌 Función para enviar recomendaciones cada mes
def enviar_recomendaciones():
    """Recorre la BD de usuarios y envía recomendaciones de inversión cada mes."""
    conn = sqlite3.connect("user_ironbrick.db")  # Cambiado a la BD correcta
    cursor = conn.cursor()
    
    cursor.execute("SELECT telegram_id FROM usuarios")
    usuarios = cursor.fetchall()
    conn.close()

    for user_id in usuarios:
        user_id = user_id[0]
        recomendaciones = obtener_recomendaciones(user_id)

        if recomendaciones is None or recomendaciones.empty:
            bot.send_message(user_id, "😞 No encontramos sets de inversión según tus criterios este mes.")
        else:
            mensaje = "📊 *Propuestas de Inversión en LEGO* 📊\n\n"
            for _, row in recomendaciones.iterrows():
                mensaje += f"🧱 *{row['SetName']}* ({row['Number']})\n"
                mensaje += f"💰 *Precio Actual:* ${row['USRetailPrice']}\n"
                mensaje += f"📈 *Rentabilidad:* {row['PredictedInvestmentScore']:.2f}\n"
                mensaje += f"🛒 *Tema:* {row['Theme']}\n"
                mensaje += f"🔗 [Más info](https://bricklink.com/{row['Number']})\n\n"

            bot.send_message(user_id, mensaje, parse_mode="Markdown")

# 📌 Ejecutar el bot una vez al mes sin bloquear el proceso
schedule.every(30).days.do(enviar_recomendaciones)

while True:
    schedule.run_pending()
    time.sleep(60 * 60 * 24)  # Revisa cada 24 horas si debe enviar alertas
