import streamlit as st
import sqlite3

# 📌 Creo conexión a la base de datos SQLite
conn = sqlite3.connect("user_ironbrick.db", check_same_thread=False)
cursor = conn.cursor()

# 📌 Creo la tabla si no existe
cursor.execute("""
CREATE TABLE IF NOT EXISTS usuarios (
    telegram_id TEXT PRIMARY KEY,
    presupuesto_max INTEGER,
    temas_favoritos TEXT,
    rentabilidad_min INTEGER,
    exclusivo TEXT
)
""")
conn.commit()

# 📌 Interfaz de Streamlit
st.title("📢 Configuración de Alertas de Inversión en LEGO")

st.write("Registra tus preferencias para recibir alertas cada 15 días en Telegram.")

telegram_id = st.text_input("🔹 Tu ID de Telegram (usa @userinfobot en Telegram para obtenerlo)")
presupuesto_max = st.number_input("💰 Presupuesto Máximo (USD)", min_value=50, value=500)
temas_favoritos = st.multiselect("🛒 Temas Favoritos", ["Star Wars", "Technic", "Creator Expert", "Harry Potter", "Ideas", "Speed Champion", "Ninjago", "Classic", "City", "BrickHeadz"])
rentabilidad_min = st.slider("📈 Rentabilidad esperada en 2 años (%)", 10, 100, 30)
exclusivo = st.checkbox("🔒 Solo sets exclusivos")

# 📌 Guardar información en la base de datos
if st.button("💾 Guardar configuración"):
    temas_str = ",".join(temas_favoritos)
    exclusivo_str = "Sí" if exclusivo else "No"
    
    cursor.execute("""
    INSERT OR REPLACE INTO usuarios (telegram_id, presupuesto_max, temas_favoritos, rentabilidad_min, piezas_min, exclusivo)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (telegram_id, presupuesto_max, temas_str, rentabilidad_min, exclusivo_str))
    
    conn.commit()
    st.success("✅ ¡Tus preferencias han sido guardadas! Recibirás alertas en Telegram.")

