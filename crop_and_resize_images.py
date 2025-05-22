import os
from PIL import Image

INPUT_FOLDER = "fotos"
OUTPUT_FOLDER = "cropped_images"
TARGET_SIZE = (224, 224)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def center_crop(image):
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    return image.crop((left, top, right, bottom))

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            path = os.path.join(INPUT_FOLDER, filename)
            image = Image.open(path).convert("RGB")
            cropped = center_crop(image)
            resized = cropped.resize(TARGET_SIZE)
            resized.save(os.path.join(OUTPUT_FOLDER, filename))
            print(f"✅ {filename} recortada y redimensionada.")
        except Exception as e:
            print(f"❌ Error con {filename}: {e}")

print("🎉 Recorte completado.")
