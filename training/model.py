import tensorflow as tf

NUM_CLASSES = 10
IMG_SIZE = (224, 224)

# Load pretrained ResNet50
base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

# Freeze everything first
base_model.trainable = True

# Freeze the earlier layers.
# Only the final ~30 layers will be fine-tuned.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Build model
inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))

# ResNet50 preprocessing
x = tf.keras.applications.resnet50.preprocess_input(inputs)

# ResNet50 feature extractor
x = base_model(x, training=False)

# Classification head
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)

# 10 tomato disease classes
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

print("\nTrainable layers:")

for layer in base_model.layers:
    if layer.trainable:
        print(layer.name)

print("\nModel summary:\n")
model.summary()