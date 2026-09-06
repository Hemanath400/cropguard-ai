import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from prepare_dataset import (
    train_ds,
    val_ds,
    class_names,
    train_labels_int
)

BEST_MODEL = "models/resnet50_tomato_best.keras"

print("\nLoading best Epoch-1 model...")
model = tf.keras.models.load_model(BEST_MODEL)

# ------------------------------------------------------------
# Fine-tune ResNet50
# ------------------------------------------------------------

base_model = model.get_layer("resnet50")

base_model.trainable = True

# Freeze all but last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# ------------------------------------------------------------
# Class weights
# ------------------------------------------------------------

classes = np.arange(len(class_names))

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.array(train_labels_int)
)

class_weights = dict(zip(classes, weights))

print("\nClass weights:")
for i, weight in class_weights.items():
    print(f"{i}: {class_names[i]} -> {weight:.4f}")

# ------------------------------------------------------------
# Compile
# ------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    ]
)

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/resnet50_tomato_finetuned_best.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=1,
    min_lr=1e-7,
    verbose=1
)

# ------------------------------------------------------------
# Fine-tuning
# ------------------------------------------------------------

print("\nStarting fine-tuning...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ------------------------------------------------------------
# Save final model
# ------------------------------------------------------------

model.save(
    "models/resnet50_tomato_finetuned_final.keras"
)

np.save(
    "models/finetuning_history.npy",
    history.history,
    allow_pickle=True
)

print("\nFine-tuning complete!")
print("Best model:")
print("models/resnet50_tomato_finetuned_best.keras")