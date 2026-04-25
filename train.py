import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_build import build_vgg16_model

# 🔹 Load the preprocessed data
with open("processed_data.pkl", "rb") as f:
    X_train, X_val, X_test, y_train, y_val, y_test, label_map = pickle.load(f)

class_names = list(label_map.keys())

# 🔹 Build model
model = build_vgg16_model(num_classes=len(class_names))

# 🔹 Callbacks
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
mc = ModelCheckpoint("plant_vgg16.h5", monitor="val_accuracy", save_best_only=True)

# 🔹 Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[es, mc]
)

# 🔹 Save history
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Training complete! Model saved as plant_vgg16.h5")
