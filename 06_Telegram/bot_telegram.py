import telebot
import sqlite3
import pandas as pd
import time
from dotenv import load_dotenv

# 📌 Token del bot de Telegram
# Configuración del entorno
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# 📌 Cargar datos de sets retirados
df_lego = pd.read_csv("../01_Data_Cleaning/df_lego_final_retirados.csv")

def obtener_recomendaciones(telegram_id):
    """Busca en la BD los requisitos del usuario y filtra los sets de LEGO."""
    conn = sqlite3.connect("user_ironbrick.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM usuarios WHERE telegram_id = ?", (telegram_id,))
    usuario = cursor.fetchone()
    conn.close()

    if not usuario:
        return None

    _, presupuesto_max, temas_favoritos, rentabilidad_min, piezas_min, exclusivo = usuario
    temas_lista = temas_favoritos.split(",")

    df_filtrado = df_lego[
        (df_lego["CurrentValueNew"] <= presupuesto_max) &
        (df_lego["Theme"].isin(temas_lista)) &
        (df_lego["ForecastValueNew2Y"] >= (df_lego["CurrentValueNew"] * (1 + rentabilidad_min / 100))) &
        (df_lego["PieceCount"] >= piezas_min)
    ]

    if exclusivo == "Sí":
        df_filtrado = df_filtrado[df_filtrado["Exclusive"] == "Yes"]

    return df_filtrado

def enviar_recomendaciones():
    """Envía recomendaciones a los usuarios cada 15 días."""
    conn = sqlite3.connect("user_ironbrick.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT telegram_id FROM usuarios")
    usuarios = cursor.fetchall()
    conn.close()

    for user_id in usuarios:
        user_id = user_id[0]
        recomendaciones = obtener_recomendaciones(user_id)

        if recomendaciones is None or recomendaciones.empty:
            bot.send_message(user_id, "😞 No encontramos sets de inversión según tus criterios esta vez. ¡Revisaremos en 15 días!")
        else:
            mensaje = "📊 *Recomendaciones de Inversión en LEGO* 📊\n\n"
            for _, row in recomendaciones.iterrows():
                mensaje += f"🧱 *{row['SetName']}* ({row['SetNumber']})\n"
                mensaje += f"💰 *Precio Actual:* ${row['CurrentValueNew']}\n"
                mensaje += f"📈 *Valor Estimado en 2 años:* ${row['ForecastValueNew2Y']}\n"
                mensaje += f"🛒 *Tema:* {row['Theme']}\n"
                mensaje += f"🔗 [Más info](https://bricklink.com/{row['SetNumber']})\n\n"

            bot.send_message(user_id, mensaje, parse_mode="Markdown")

# 📌 Bucle para enviar alertas cada 15 días
while True:
    enviar_recomendaciones()
    time.sleep(60 * 60 * 24 * 15)  # Esperar 15 días antes de volver a enviar
