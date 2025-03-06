from pymongo import MongoClient

# 📌 Reemplaza con tu URI correcta
uri = "mongodb+srv://erove:ironbrick@cluster0.b1drs.mongodb.net/?retryWrites=true&w=majority"  # O tu conexión Atlas

try:
    client = MongoClient(uri)
    db = client["dbo_lego"]  # Cambia esto por el nombre de tu base de datos
    print("✅ Conexión exitosa a MongoDB")
    print("📌 Bases de datos disponibles:", client.list_database_names())
except Exception as e:
    print("❌ Error de conexión:", e)


# Listar las colecciones en tu base de datos
collections = db.list_collection_names()
print("📌 Colecciones disponibles:", collections)
