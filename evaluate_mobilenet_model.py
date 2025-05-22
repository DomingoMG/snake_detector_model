import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from tensorflow.keras.preprocessing import image

# Configuración
VAL_DIR = "data/val"
MODEL_PATH = "snake_classifier_mobilenet.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
UMBRAL = 0.76

# Cargar modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Generador para validación
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Evaluar
loss, accuracy = model.evaluate(val_generator)
print(f"\n📉 Pérdida (Loss): {loss:.4f}")
print(f"✅ Precisión (Accuracy): {accuracy * 100:.2f}%")

# Predicciones
val_generator.reset()
predictions = model.predict(val_generator)
y_pred = (predictions > UMBRAL).astype("int32").flatten()
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())
file_paths = val_generator.filepaths

# Reporte
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

# # Mostrar imágenes mal clasificadas como snake (falsos positivos)
# false_positives_idx = np.where((y_true == 0) & (y_pred == 1))[0]
# print(f"\n🔍 Mostrando {len(false_positives_idx)} imágenes mal clasificadas como 'snake'")

# plt.figure(figsize=(18, 6))
# for i, idx in enumerate(false_positives_idx[:6]):
#     img_path = file_paths[idx]
#     img = image.load_img(img_path, target_size=IMG_SIZE)
#     pred_score = predictions[idx][0]
    
#     plt.subplot(1, 6, i + 1)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(f"Score: {pred_score:.2f}")

# plt.suptitle("Falsos positivos: no_snake → snake", fontsize=14)
# plt.tight_layout()
# plt.show()
