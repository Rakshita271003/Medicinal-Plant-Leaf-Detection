import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

IMG_SIZE = 224

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):  # ✅ only process files, not subfolders
            img = cv2.imread(file_path)
            if img is None:
                print(f"⚠️ Skipping unreadable file: {file_path}")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return images, labels

def preprocess_and_save(dataset_dir):
    categories = os.listdir(os.path.join(dataset_dir, "Train"))  # class names
    label_map = {cat: idx for idx, cat in enumerate(categories)}

    images, labels = [], []
    for category in categories:
        folder = os.path.join(dataset_dir, "Train", category)
        imgs, lbls = load_images_from_folder(folder, label_map[category])
        images.extend(imgs)
        labels.extend(lbls)

    X = np.array(images) / 255.0
    y = np.array(labels)

    # split train/test/validate
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # save
    with open("processed_data.pkl", "wb") as f:
        pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test, label_map), f)

    print("✅ Data preprocessed and saved as processed_data.pkl")

if __name__ == "__main__":
    preprocess_and_save("data")
