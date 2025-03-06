import os
import psycopg2
import telebot
import time

# 📌 Obtener el token del bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# 📌 URL de la base de datos PostgreSQL en Render
DB_URL = os.getenv("DATABASE_URL", "postgresql://ironbrick_user:password@your-database-host.compute.amazonaws.com:5432/ironbrick")

# 📌 Función para conectar a la base de datos PostgreSQL en Render
def get_db_connection():
    return psycopg2.connect(DB_URL, sslmode="require")

@bot.message_handler(commands=['status'])
def check_status(message):
    user_id = message.chat.id
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios WHERE telegram_id = %s", (user_id,))
    usuario = cursor.fetchone()

    if usuario:
        presupuesto_min, presupuesto_max, temas_favoritos = usuario
        mensaje = f"📊 *Estado de tu suscripción:*\n\n"
        mensaje += f"💰 *Presupuesto:* ${presupuesto_min} - ${presupuesto_max}\n"
        mensaje += f"🛒 *Temas favoritos:* {temas_favoritos}\n"
        mensaje += "✅ Tu suscripción está activa y funcionando correctamente."
    else:
        mensaje = "❌ No estás registrado en el sistema. Usa `/start` para suscribirte."

    conn.close()
    bot.send_message(user_id, mensaje, parse_mode="Markdown")

# 📌 Iniciar el bot
if __name__ == "__main__":
    print("🔄 Iniciando bot en modo seguro...")
    while True:
        try:
            bot.infinity_polling(timeout=60, long_polling_timeout=10)
        except Exception as e:
            print(f"⚠️ Error en el bot: {e}")
            time.sleep(5)
