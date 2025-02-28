import requests
import os
from dotenv import load_dotenv

# Cargamos variables de entorno
load_dotenv()

# Obtenemos las credenciales de las variables del entorno
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_alert(set_name, predicted_2y, predicted_5y):
    """
    Envío de una alerta por Telegram si un set de LEGO tiene alta probabilidad de revalorización.
    """
    message = (
        f"📢 *Alerta de Inversión LEGO* 📢\n\n"
        f"🔹 *Set:* {set_name}\n"
        f"📈 *Predicción a 2 años:* ${predicted_2y:.2f}\n"
        f"📈 *Predicción a 5 años:* ${predicted_5y:.2f}\n"
        f"💡 ¡Considera invertir en este set antes de que suba de precio!"
    )
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=data)
