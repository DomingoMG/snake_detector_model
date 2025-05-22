import os
from PIL import Image

folder_path = r'mangueras' 
min_width = 224
min_height = 224

if not os.path.exists(folder_path):
    print(f"❌ La ruta no existe: {folder_path}")
    exit(1)

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        with Image.open(file_path) as img:
            img.load()  
            width, height = img.size

        if width < min_width or height < min_height:
            print(f"🗑️ Eliminando {filename} ({width}x{height})")
            os.remove(file_path)

    except Exception as e:
        print(f"⚠️ No se pudo procesar {filename}: {e}")
