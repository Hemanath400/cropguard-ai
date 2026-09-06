import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from prepare_dataset import (
    train_ds,
    val_ds,
    class_names,
    train_labels_int
)

from model import model


# ============================================================
# 1. LOAD THE BEST MODEL FROM EPOCH 1
# ============================================================

BEST_MODEL = "models/resnet50_tomato_best.keras"

print("\nLoading best Epoch-1 model...")

model = tf.keras.models.load_model(BEST_MODEL)

print("Best model loaded successfully.")


# ============================================================
# 2. UNFREEZE ONLY THE LAST 30 RESNET50 LAYERS
# ============================================================

base_model = model.get_layer("resnet50")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

print("\nFine-tuning configuration:")
print("Frozen ResNet50 layers:", len(base_model.layers) - 30)
print("Trainable ResNet50 layers: 30")


# ============================================================
# 3. CREATE CLASS WEIGHTS
# ============================================================

classes = np.arange(len(class_names))

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.array(train_labels_int)
)

class_weights = dict(zip(classes, weights))

print("\nClass weights:")

for class_index, weight in class_weights.items():
    print(f"{class_index}: {class_names[class_index]} -> {weight:.4f}")


# ============================================================
# 4. COMPILE WITH A VERY LOW LEARNING RATE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    ]
)


# ============================================================
# 5. CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/resnet50_tomato_finetuned_best.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 6. FINE-TUNE
# ============================================================

print("\nStarting fine-tuning...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ============================================================
# 7. SAVE FINE-TUNED MODEL
# ============================================================

model.save(
    "models/resnet50_tomato_finetuned_final.keras"
)

np.save(
    "models/finetuning_history.npy",
    history.history,
    allow_pickle=True
)

print("\nFine-tuning completed!")
print(
    "Best fine-tuned model: "
    "models/resnet50_tomato_finetuned_best.keras"
)