import telebot
import sqlite3
import pandas as pd
import joblib
import time

# 📌 Token del bot de Telegram
TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# 📌 Cargar el modelo
modelo_path = "/mnt/data/stacking_model.pkl"
modelo = joblib.load(modelo_path)

# 📌 Cargar dataset de sets en venta
df_lego = pd.read_csv("/mnt/data/df_lego_final_venta.csv")

# 📌 Aplicar el modelo para predecir oportunidades de inversión
df_lego["PredictedInvestmentScore"] = modelo.predict(df_lego[['USRetailPrice', 'Pieces', 'Minifigs', 'YearsSinceExit', 
                                                               'ResaleDemand', 'AnnualPriceIncrease', 'Exclusivity', 
                                                               'SizeCategory', 'PricePerPiece', 'PricePerMinifig', 'YearsOnMarket']])

# 📌 Función para obtener recomendaciones personalizadas
def obtener_recomendaciones(telegram_id):
    """Filtra sets según los criterios del usuario en la BD."""
    conn = sqlite3.connect("usuarios.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM usuarios WHERE telegram_id = ?", (telegram_id,))
    usuario = cursor.fetchone()
    conn.close()

    if not usuario:
        return None

    _, presupuesto_max, temas_favoritos, rentabilidad_min, piezas_min, exclusivo = usuario
    temas_lista = temas_favoritos.split(",")

    df_filtrado = df_lego[
        (df_lego["USRetailPrice"] <= presupuesto_max) &
        (df_lego["Theme"].isin(temas_lista)) &
        (df_lego["PredictedInvestmentScore"] >= rentabilidad_min) &
        (df_lego["Pieces"] >= piezas_min)
    ]

    if exclusivo == "Sí":
        df_filtrado = df_filtrado[df_filtrado["Exclusivity"] == 1]

    return df_filtrado.sort_values(by="PredictedInvestmentScore", ascending=False).head(5)

# 📌 Función para enviar recomendaciones cada mes
def enviar_recomendaciones():
    """Recorre la BD de usuarios y envía recomendaciones de inversión cada mes."""
    conn = sqlite3.connect("usuarios.db")
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
                mensaje += f"🧱 *{row['SetName']}* ({row['SetNumber']})\n"
                mensaje += f"💰 *Precio Actual:* ${row['USRetailPrice']}\n"
                mensaje += f"📈 *Investment Score:* {row['PredictedInvestmentScore']:.2f}\n"
                mensaje += f"🛒 *Tema:* {row['Theme']}\n"
                mensaje += f"🔗 [Más info](https://bricklink.com/{row['SetNumber']})\n\n"

            bot.send_message(user_id, mensaje, parse_mode="Markdown")

# 📌 Ejecutar el bot cada 30 días (una vez al mes)
while True:
    enviar_recomendaciones()
    time.sleep(60 * 60 * 24 * 30)
