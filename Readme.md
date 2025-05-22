# 🐍 snake_detector_model
Python project for training a binary image classification model to detect the presence of snakes — specifically the invasive **California kingsnake** (*Lampropeltis californiae*) — to support invasive species monitoring efforts in the Canary Islands.
This model is used by the `snake_detector_api` to predict whether an image contains a snake or not.
---

## 🧠 Model
Uses **MobileNetV2** as a pretrained feature extractor combined with custom dense layers for binary classification (`snake` vs `no_snake`).

Training is done using:
- `TensorFlow` / `Keras`
- Image size: 224x224
- `binary_crossentropy` loss and `Adam` optimizer
---

## 📁 Repository Structure
| File | Description |
|------|-------------|
| `keras_tensorflow_mobilenet_model.py` | Trains the MobileNetV2-based snake detection model. |
| `evaluate_mobilenet_model.py` | Evaluates the trained model with confusion matrix and classification report. |
| `crop_and_resize_images.py` | Crops and resizes images to 224x224 px. |
| `remove_small_images.py` | Removes images smaller than the target dimensions. |
| `rotated_images.py` | Generates rotated images for data augmentation. |
| `filter_hose_images_blip.py` | Filters out irrelevant images (e.g., hoses) using BLIP-based VQA. |
---

## 🗃️ Expected Dataset Structure
```
data/
├── train/
│   ├── snake/
│   └── no_snake/
└── val/
    ├── snake/
    └── no_snake/
```
- Ensure images are centered, sharp, and at least 224x224 pixels.
- Use the included preprocessing scripts to clean, crop, and augment the dataset.
---

## 🚀 Training
```bash
python keras_tensorflow_mobilenet_model.py
```
This saves the trained model as `snake_classifier_mobilenet.h5`.
---

## 📈 Evaluation
```bash
python evaluate_mobilenet_model.py
```

Includes:
- Loss and accuracy
- Confusion matrix
- Classification report
- (Optional) Visualization of false positives
---

## 🔍 Image Filtering with BLIP
Use `filter_hose_images_blip.py` to automatically discard non-relevant images (like hoses) that may be visually similar to snakes.
---

## 📦 Requirements
```bash
pip install tensorflow pillow matplotlib seaborn scikit-learn transformers
```

---

## 📜 License
This project is licensed under the MIT License.  
Contribute to improve the model and help reduce the impact of invasive species 🐍🌿.
---

## 👨‍💻 Author
Developed by [DomingoMG](https://github.com/DomingoMG) as part of an AI-assisted environmental conservation project.
