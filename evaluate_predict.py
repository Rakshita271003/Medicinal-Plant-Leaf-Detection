import numpy as np
import tensorflow as tf

# Load test data
X_test = np.load("test_images.npy")
y_test = np.load("test_labels.npy")
class_names = np.load("class_names.npy", allow_pickle=True)

# Load model
model = tf.keras.models.load_model("models/medicinal_leaf_model.h5")

loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
