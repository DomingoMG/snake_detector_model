import os
import random
from PIL import Image

# Rutas
INPUT_FOLDER = 'cropped_images'
OUTPUT_FOLDER = 'rotated_mangueras'

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Rango de rotación (grados)
MIN_ROTATION = -15
MAX_ROTATION = 15

# Procesar imágenes
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        input_path = os.path.join(INPUT_FOLDER, filename)

        try:
            # Abrir imagen original
            image = Image.open(input_path)
            base_name, ext = os.path.splitext(filename)

            # Guardar imagen original en la nueva carpeta
            image.save(os.path.join(OUTPUT_FOLDER, filename))

            # Rotación aleatoria
            rotation_angle = random.uniform(90, 90)
            rotated_image = image.rotate(rotation_angle, expand=True, fillcolor=(255, 255, 255))

            # Guardar imagen rotada
            rotated_filename = f"{base_name}_rotated{ext}"
            rotated_image.save(os.path.join(OUTPUT_FOLDER, rotated_filename))

            print(f"✅ Procesada: {filename} -> Rotada {rotation_angle:.2f}°")

        except Exception as e:
            print(f"❌ Error procesando {filename}: {e}")
