import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pickle


model = load_model("plant_vgg16.h5")

with open("processed_data.pkl", "rb") as f:
    _, _, _, _, _, _, label_map = pickle.load(f)


class_names = {v: k for k, v in label_map.items()}

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Cannot read image: {image_path}")
        return
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0 

    pred = model.predict(img)
    class_idx = np.argmax(pred)
    print(f"Predicted class: {class_names[class_idx]}")

# 🔹 Test on a sample image
predict_image("data/Test/Rose/IMG_20201003_172258.jpg")
