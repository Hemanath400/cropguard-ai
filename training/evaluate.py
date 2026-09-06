import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from prepare_dataset import test_ds, class_names

MODEL_PATH = "models/resnet50_tomato_finetuned_best.keras"

print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

print("\nRunning full test evaluation...")

# Get all test images and labels
all_images = []
all_labels = []

for images, labels in test_ds:
    all_images.append(images)
    all_labels.append(labels)

X_test = tf.concat(all_images, axis=0)
y_true = tf.concat(all_labels, axis=0).numpy()

print("Test images:", X_test.shape)
print("Test labels:", y_true.shape)

# Predict everything
predictions = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

cm = confusion_matrix(y_true, y_pred)

print(cm)

print("\nClass order:")

for i, name in enumerate(class_names):
    print(i, "->", name)