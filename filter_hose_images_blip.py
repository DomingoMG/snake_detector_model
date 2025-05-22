import os
import shutil
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering

# Configuración de carpetas
INPUT_FOLDER = "mangueras"
OUTPUT_FOLDER = "filtered_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Dispositivo: GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo BLIP VQA
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# Pregunta
QUESTION = "Is there a hose on a white background? YES OR NO"

# Procesar imágenes
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(INPUT_FOLDER, filename)
        try:
            raw_image = Image.open(path).convert("RGB")

            inputs = processor(raw_image, QUESTION, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            answer = processor.decode(out[0], skip_special_tokens=True)

            print(f"{filename} ➜ {answer}")

            # Si la respuesta sugiere que está en un entorno natural, la conservamos
            if any(x in answer.lower() for x in ["no", "NO"]):
                shutil.copy(path, os.path.join(OUTPUT_FOLDER, filename))

        except Exception as e:
            print(f"❌ Error con {filename}: {e}")

print("✅ Filtrado completado.")
