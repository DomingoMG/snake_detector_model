import tensorflow as tf


ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
layers = tf.keras.layers
models = tf.keras.models

# Configuración
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'

# Preprocesamiento de imágenes
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Cargar MobileNetV2 como base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

# Opcional: congelar capas base (para evitar que se reentrene MobileNet)
base_model.trainable = False

# Añadir tus propias capas de clasificación
model = models.Sequential([
    base_model,
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # binaria
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Guardar modelo
model.save("snake_classifier_mobilenet.h5")
