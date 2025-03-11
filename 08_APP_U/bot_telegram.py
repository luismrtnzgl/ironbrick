import os
import psycopg2
import telebot
import joblib
import requests
import pandas as pd
import numpy as np
import schedule
import time

# Obtenemos el token del bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# URL de la base de datos PostgreSQL en Render
DB_URL = os.getenv("DATABASE_URL")

# URL del modelo de predicción en GitHub
modelo_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/05_Streamlit/models/stacking_model.pkl"

# URL del dataset de LEGO en GitHub
dataset_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/01_Data_Cleaning/df_lego_final_venta.csv"

# Función para conectar a la base de datos PostgreSQL en Render
def get_db_connection():
    return psycopg2.connect(DB_URL, sslmode="require")

# Cargamos el modelo de predicción
def load_model():
    modelo_path = "/tmp/stacking_model.pkl"
    
    if not os.path.exists(modelo_path):
        response = requests.get(modelo_url)
        with open(modelo_path, "wb") as f:
            f.write(response.content)
    
    return joblib.load(modelo_path)

modelo = load_model()

# Cargamos y procesamos el dataset de LEGO
def load_data():
    df = pd.read_csv(dataset_url)
    return preprocess_data(df)

def preprocess_data(df):
    df = df[df['USRetailPrice'] > 0].copy()

    exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
    df['Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)

    size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
    df['SizeCategory'] = df['SizeCategory'].map(size_category_mapping)

    df["PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]
    df["PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)
    df["YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

df_lego = load_data()

# Función para obtener el mejor set sin repetir recomendaciones
def obtener_nueva_recomendacion(telegram_id, presupuesto_min, presupuesto_max, temas_favoritos):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT set_id FROM recomendaciones WHERE telegram_id = %s", (str(telegram_id),))
    sets_recomendados = {row[0] for row in cursor.fetchall()}

    df_filtrado = df_lego[(df_lego["USRetailPrice"] >= presupuesto_min) & 
                           (df_lego["USRetailPrice"] <= presupuesto_max)]

    if "Todos" not in temas_favoritos:
        df_filtrado = df_filtrado[df_filtrado["Theme"].isin(temas_favoritos)]

    df_filtrado = df_filtrado[~df_filtrado["Number"].astype(str).isin(sets_recomendados)]

    if df_filtrado.empty:
        return None

    features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 'ResaleDemand', 
                'AnnualPriceIncrease', 'Exclusivity', 'SizeCategory', 'PricePerPiece', 
                'PricePerMinifig', 'YearsOnMarket']
    
    for col in features:
        if col not in df_filtrado.columns:
            df_filtrado[col] = 0  

    df_filtrado["PredictedInvestmentScore"] = modelo.predict(df_filtrado[features])

    return df_filtrado.sort_values(by="PredictedInvestmentScore", ascending=False).iloc[0]

# Función para enviar recomendación a todos los usuarios registrados (mensual)
def enviar_recomendaciones():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT telegram_id, presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios")
    usuarios = cursor.fetchall()
    
    for user in usuarios:
        user_id, presupuesto_min, presupuesto_max, temas_favoritos = user
        temas_favoritos = temas_favoritos.split(",")

        mejor_set = obtener_nueva_recomendacion(user_id, presupuesto_min, presupuesto_max, temas_favoritos)

        if mejor_set is not None:
            mensaje = f"📊 *Nueva Oportunidad de Inversión en LEGO*\n\n"
            mensaje += f"🧱 *{mejor_set['SetName']}* ({mejor_set['Number']})\n"
            mensaje += f"💰 *Precio:* ${mejor_set['USRetailPrice']:.2f}\n"
            mensaje += f"📈 *Rentabilidad Estimada:* {mejor_set['PredictedInvestmentScore']:.2f}\n"
            mensaje += f"🛒 *Tema:* {mejor_set['Theme']}\n"
            mensaje += f"🔗 [Ver en BrickLink](https://www.bricklink.com/v2/catalog/catalogitem.page?S={mejor_set['Number']})\n"

            bot.send_message(user_id, mensaje, parse_mode="Markdown")
        else:
            bot.send_message(user_id, "😞 No encontramos sets adecuados en tu rango de presupuesto y temas seleccionados.")

    conn.close()

# Función para confirmar la inscripción
def confirmar_suscripcion(telegram_id):
    """Envia un mensaje de confirmación al usuario con sus datos de suscripción."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios WHERE telegram_id = %s", (telegram_id,))
    usuario = cursor.fetchone()

    if usuario:
        presupuesto_min, presupuesto_max, temas_favoritos = usuario
        mensaje = (f"📢 *¡Hemos recibido tu suscripción!* 🎉\n\n"
                   f"💰 *Rango de precios:* ${presupuesto_min} - ${presupuesto_max}\n"
                   f"🛒 *Temas favoritos:* {temas_favoritos}\n\n"
                   "🔔 Recibirás recomendaciones de inversión en LEGO según estas preferencias.")
        bot.send_message(telegram_id, mensaje, parse_mode="Markdown")

    conn.close()

# Función para clasificar la rentabilidad en categorías
def clasificar_revalorizacion(score):
    if score > 13:
        return "Muy Alta"
    elif 10 <= score <= 13:
        return "Alta"
    elif 5 <= score < 10:
        return "Media"
    elif 0 <= score < 5:
        return "Baja"
    else:
        return "Ninguna"

# Función para enviar recomendación manual a un usuario específico
def enviar_recomendacion_manual(telegram_id):
    print(f"🔹 Enviando recomendación manual a {telegram_id}...")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios WHERE telegram_id = %s", (str(telegram_id),))
    usuario = cursor.fetchone()

    if usuario:
        presupuesto_min, presupuesto_max, temas_favoritos = usuario
        temas_favoritos = temas_favoritos.split(",")

        mejor_set = obtener_nueva_recomendacion(telegram_id, presupuesto_min, presupuesto_max, temas_favoritos)

        if mejor_set is not None:
            mensaje = f"📊 *Nueva Oportunidad de Inversión en LEGO*\n\n"
            mensaje += f"🧱 *{mejor_set['SetName']}* ({mejor_set['Number']})\n"
            mensaje += f"💰 *Precio:* ${mejor_set['USRetailPrice']:.2f}\n"
            mensaje += f"📈 *Rentabilidad Estimada:* {clasificar_revalorizacion(mejor_set['PredictedInvestmentScore'])}\n"
            mensaje += f"🛒 *Tema:* {mejor_set['Theme']}\n"
            mensaje += f"🔗 [Ver en BrickLink](https://www.bricklink.com/v2/catalog/catalogitem.page?S={mejor_set['Number']})\n"

            bot.send_message(user_id, mensaje, parse_mode="Markdown")
        else:
            bot.send_message(telegram_id, "😞 No encontramos sets adecuados en tu rango de presupuesto y temas seleccionados.")
    
    else:
        print(f"❌ No se encontró al usuario con ID {telegram_id} en la base de datos.")
    
    conn.close()

# Manejo del comando /start
@bot.message_handler(commands=['start'])
def start(message):
    telegram_id = str(message.chat.id)
    conn = get_db_connection()
    cursor = conn.cursor()

    # Verificar si el usuario ya está registrado
    cursor.execute("SELECT * FROM usuarios WHERE telegram_id = %s", (telegram_id,))
    usuario = cursor.fetchone()

    if usuario:
        bot.send_message(telegram_id, "✅ ¡Ya estás registrado en el sistema de alertas de inversión en LEGO!")
    else:
        # Registrar al usuario con valores por defecto
        cursor.execute("""
            INSERT INTO usuarios (telegram_id, presupuesto_min, presupuesto_max, temas_favoritos) 
            VALUES (%s, %s, %s, %s)
        """, (telegram_id, 10, 200, 'Todos'))
        conn.commit()
        bot.send_message(telegram_id, "🎉 ¡Bienvenido al sistema de alertas de inversión en LEGO! "
                                      "Te hemos registrado con un rango de precios de $10 a $200 y todos los temas. "
                                      "Puedes modificar tus preferencias en la web de Streamlit.")

    conn.close()

# Manejo del comando /status
@bot.message_handler(commands=['status'])
def status(message):
    telegram_id = str(message.chat.id)
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT presupuesto_min, presupuesto_max, temas_favoritos FROM usuarios WHERE telegram_id = %s", (telegram_id,))
    usuario = cursor.fetchone()

    if usuario:
        presupuesto_min, presupuesto_max, temas_favoritos = usuario
        mensaje = (f"📊 *Estado de tu suscripción:*\n\n"
                   f"💰 *Rango de precios:* ${presupuesto_min} - ${presupuesto_max}\n"
                   f"🛒 *Temas favoritos:* {temas_favoritos}\n\n"
                   "Puedes modificar tus preferencias en la web de Streamlit.")
        bot.send_message(telegram_id, mensaje, parse_mode="Markdown")
    else:
        bot.send_message(telegram_id, "⚠️ No estás registrado en el sistema. Escribe /start para registrarte.")

    conn.close()

# Programar el envío cada 30 días
schedule.every(30).days.do(enviar_recomendaciones)

# Iniciar el bot y el sistema de alertas
if __name__ == "__main__":
    print("🔄 Iniciando bot con alertas de inversión...")

    import threading
    def run_scheduler():
        while True:
            try:
                schedule.run_pending()
                time.sleep(86400)  
            except Exception as e:
                print(f"⚠️ Error en el sistema de alertas: {e}")
                time.sleep(60)  

    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    while True:
        try:
            bot.infinity_polling(timeout=60, long_polling_timeout=10)
        except telebot.apihelper.ApiTelegramException as e:
            if e.error_code == 409:  
                print("⚠️ Se detectó una segunda instancia del bot. Cerrando esta para evitar conflictos.")
                break  
        except Exception as e:
            print(f"⚠️ Error en el bot: {e}")
            time.sleep(60)  
