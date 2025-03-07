import os
import psycopg2
import telebot
import joblib
import requests
import pandas as pd
import numpy as np
import schedule
import time

# 📌 Obtener el token del bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# 📌 URL de la base de datos PostgreSQL en Render
DB_URL = os.getenv("DATABASE_URL")

# 📌 URL del modelo de predicción en GitHub
modelo_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/05_Streamlit/models/stacking_model.pkl"

# 📌 URL del dataset de LEGO en GitHub
dataset_url = "https://raw.githubusercontent.com/luismrtnzgl/ironbrick/main/01_Data_Cleaning/df_lego_final_venta.csv"

# 📌 Función para conectar a la base de datos PostgreSQL en Render
def get_db_connection():
    return psycopg2.connect(DB_URL, sslmode="require")

# 📌 Cargar el modelo de predicción
def load_model():
    modelo_path = "/tmp/stacking_model.pkl"
    
    if not os.path.exists(modelo_path):
        response = requests.get(modelo_url)
        with open(modelo_path, "wb") as f:
            f.write(response.content)
    
    return joblib.load(modelo_path)

modelo = load_model()

# 📌 Cargar y procesar el dataset de LEGO
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

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    return df

df_lego = load_data()

# 📌 Asegurar que las tablas `usuarios` y `recomendaciones` existen
def crear_tablas():
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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recomendaciones (
        telegram_id TEXT,
        set_id TEXT,
        PRIMARY KEY (telegram_id, set_id)
    )
    """)

    conn.commit()
    conn.close()

crear_tablas()

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

# 📌 Función para obtener el mejor set que aún no ha sido recomendado
def obtener_nueva_recomendacion(telegram_id, presupuesto_min, presupuesto_max, temas_favoritos):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 📌 Obtener sets que ya han sido recomendados al usuario
    cursor.execute("SELECT set_id FROM recomendaciones WHERE telegram_id = %s", (str(telegram_id),))
    sets_recomendados = {row[0] for row in cursor.fetchall()}
    
    # 📌 Filtrar sets dentro del presupuesto
    df_filtrado = df_lego[(df_lego["USRetailPrice"] >= presupuesto_min) & 
                           (df_lego["USRetailPrice"] <= presupuesto_max)]

    # 📌 Filtrar por temas favoritos si el usuario ha seleccionado alguno
    if "Todos" not in temas_favoritos:
        df_filtrado = df_filtrado[df_filtrado["Theme"].isin(temas_favoritos)]

    # 📌 Excluir sets que ya fueron recomendados
    df_filtrado = df_filtrado[~df_filtrado["Number"].astype(str).isin(sets_recomendados)]

    if df_filtrado.empty:
        return None

    # 📌 Aplicar el modelo de predicción
    features = ['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 
                'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity', 
                'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']

    df_filtrado["PredictedInvestmentScore"] = modelo.predict(df_filtrado[features])

    # 📌 Seleccionar el mejor set basado en la rentabilidad
    mejor_set = df_filtrado.sort_values(by="PredictedInvestmentScore", ascending=False).iloc[0]

    return mejor_set

# 📌 Función para enviar alertas automáticas de inversión
def enviar_recomendaciones():
    print("📢 Enviando recomendaciones a los usuarios...")
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
            mensaje += f"🛒 *Tema:* {mejor_set['Theme']}\n"

            bot.send_message(user_id, mensaje, parse_mode="Markdown")

            # 📌 Registrar la recomendación
            cursor.execute("INSERT INTO recomendaciones (telegram_id, set_id) VALUES (%s, %s)", (str(user_id), mejor_set['Number']))
            conn.commit()

    conn.close()
