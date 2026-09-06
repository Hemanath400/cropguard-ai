import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from prepare_dataset import test_ds, class_names

MODEL_PATH = "models/resnet50_tomato_finetuned_best.keras"

# Load best saved model
model = tf.keras.models.load_model(MODEL_PATH)

print("\nEvaluating best model on test set...\n")

# Test loss and accuracy
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4%}")


# Collect true labels and predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))


# Classification report
print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)