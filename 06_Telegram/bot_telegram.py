import telebot
import sqlite3
import pandas as pd
import joblib
import os
import requests
import time
import schedule
import numpy as np

# 📌 Obtener el token del bot desde las variables de entorno
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("❌ ERROR: No se encontró el TOKEN del bot de Telegram.")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# 📌 Crear la tabla recomendaciones si no existe
def asegurar_tablas():
    conn = sqlite3.connect("user_ironbrick.db")
    cursor = conn.cursor()

    # Crear la tabla de usuarios si no existe
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        telegram_id TEXT PRIMARY KEY,
        presupuesto_min INTEGER DEFAULT 10,
        presupuesto_max INTEGER DEFAULT 200,
        temas_favoritos TEXT DEFAULT 'Todos'
    )
    """)

    # Crear la tabla de recomendaciones si no existe
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recomendaciones (
        telegram_id TEXT,
        set_id TEXT,
        PRIMARY KEY (telegram_id, set_id),
        FOREIGN KEY (telegram_id) REFERENCES usuarios (telegram_id)
    )
    """)

    conn.commit()
    conn.close()

# 📌 Asegurar que las tablas existen antes de que el bot se ejecute
asegurar_tablas()

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

# 📌 Función para `/start`
@bot.message_handler(commands=['start'])
def send_welcome(message):
    user_id = message.chat.id
    bot.send_message(user_id, "✅ ¡Bienvenido! Te has registrado para recibir recomendaciones de inversión en LEGO.")

    conn = sqlite3.connect("user_ironbrick.db")
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        telegram_id TEXT PRIMARY KEY,
        presupuesto_min INTEGER DEFAULT 10,
        presupuesto_max INTEGER DEFAULT 200,
        temas_favoritos TEXT DEFAULT 'Todos'
    )
    """)
    
    cursor.execute("INSERT OR IGNORE INTO usuarios (telegram_id) VALUES (?)", (user_id,))
    
    conn.commit()
    conn.close()

    bot.send_message(user_id, "✅ *Confirmación:* Tu suscripción ha sido registrada correctamente. 📩\n\n"
                              "💡 Recibirás recomendaciones cada mes.\n"
                              "📊 Puedes configurar tu presupuesto y temas favoritos en la app de Streamlit.\n"
                              "🔍 Usa `/status` para verificar tu suscripción.", parse_mode="Markdown")

# 📌 Conectar a la misma base de datos que usa Streamlit
DB_PATH = "user_ironbrick.db" 

# Función para '/status'
@bot.message_handler(commands=['status'])
def check_status(message):
    user_id = message.chat.id
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios WHERE telegram_id = ?", (user_id,))
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

    conn.close()
    bot.send_message(user_id, mensaje, parse_mode="Markdown")

# 📌 Programar envío de recomendaciones
def enviar_recomendaciones():
    print("📤 Enviando recomendaciones a los usuarios...")
    
    conn = sqlite3.connect("user_ironbrick.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT telegram_id, presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios")
    usuarios = cursor.fetchall()
    
    for user in usuarios:
        user_id, presupuesto_min, presupuesto_max, temas_favoritos = user
        temas_favoritos = temas_favoritos.split(",")

        df_filtrado = df_lego[
            (df_lego["USRetailPrice"] >= presupuesto_min) &
            (df_lego["USRetailPrice"] <= presupuesto_max)
        ]

        if "Todos" not in temas_favoritos:
            df_filtrado = df_filtrado[df_filtrado["Theme"].isin(temas_favoritos)]

        if not df_filtrado.empty:
            mejor_set = df_filtrado.sort_values(by="PredictedInvestmentScore", ascending=False).iloc[0]
            mensaje = f"📊 *Nueva Oportunidad de Inversión en LEGO*\n\n"
            mensaje += f"🧱 *{mejor_set['SetName']}* ({mejor_set['Number']})\n"
            mensaje += f"💰 *Precio:* ${mejor_set['USRetailPrice']:.2f}\n"
            mensaje += f"📈 *Rentabilidad:* {mejor_set['PredictedInvestmentScore']:.2f}\n"
            mensaje += f"🔗 [Ver en BrickLink](https://bricklink.com/v2/catalog/catalogitem.page?S={mejor_set['Number']})\n"

            bot.send_message(user_id, mensaje, parse_mode="Markdown")

    conn.close()

schedule.every(30).days.do(enviar_recomendaciones)

# 📌 Bucle principal con manejo de errores
while True:
    try:
        if __name__ == "__main__":
            print("🔄 Iniciando bot en modo seguro...")
            bot.infinity_polling(timeout=60, long_polling_timeout=10)
    except Exception as e:
        print(f"⚠️ Error en el bot: {e}")
        time.sleep(5)
