import os
import psycopg2
import telebot

# 📌 Obtener el token del bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# 📌 URL de la base de datos PostgreSQL en Render
DB_URL = os.getenv("DATABASE_URL", "postgresql://ironbrick_user:password@your-database-host.compute.amazonaws.com:5432/ironbrick")

# 📌 Función para conectar a la base de datos PostgreSQL en Render
def get_db_connection():
    return psycopg2.connect(DB_URL, sslmode="require")

# 📌 Función para asegurar que la tabla `usuarios` existe
def crear_tabla_usuarios():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        telegram_id TEXT PRIMARY KEY,
        presupuesto_min INTEGER DEFAULT 10,
        presupuesto_max INTEGER DEFAULT 200,
        temas_favoritos TEXT DEFAULT 'Todos'
    )
    """)
    
    conn.commit()
    conn.close()

# 📌 Asegurar que la tabla `usuarios` existe al iniciar el bot
crear_tabla_usuarios()

@bot.message_handler(commands=['status'])
def check_status(message):
    user_id = message.chat.id
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios WHERE telegram_id = %s", (str(user_id),))
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
