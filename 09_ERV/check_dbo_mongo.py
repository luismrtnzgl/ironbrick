from pymongo import MongoClient

# 📌 Reemplaza con tu URI correcta
uri = "mongodb+srv://erove:ironbrick@cluster0.b1drs.mongodb.net/?retryWrites=true&w=majority"

try:
    client = MongoClient(uri)
    db = client["dbo_lego"]  # Cambia esto por el nombre de tu base de datos
    print("✅ Conexión exitosa a MongoDB")
    print("📌 Bases de datos disponibles:", client.list_database_names())
except Exception as e:
    print("❌ Error de conexión:", e)
    exit()

# Listar las colecciones en la base de datos
collections = db.list_collection_names()
print("📌 Colecciones disponibles:", collections)

# Función para analizar los tipos de datos en una colección
def analizar_tipos_de_datos(coleccion_nombre):
    coleccion = db[coleccion_nombre]
    ejemplo = coleccion.find_one()  # Obtener un documento de muestra
    if not ejemplo:
        print(f"⚠️ La colección '{coleccion_nombre}' está vacía.")
        return

    tipos = {}
    for campo, valor in ejemplo.items():
        tipos[campo] = type(valor).__name__  # Obtener el tipo de dato como string

    print(f"\n📊 Tipos de datos en la colección '{coleccion_nombre}':")
    for campo, tipo in tipos.items():
        print(f"   - {campo}: {tipo}")

# Analizar las colecciones específicas
for coleccion in ["lego_final_venta", "lego_final_retirados"]:
    if coleccion in collections:
        analizar_tipos_de_datos(coleccion)
    else:
        print(f"⚠️ La colección '{coleccion}' no existe en la base de datos.")
