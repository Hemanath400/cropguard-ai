from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split


# ============================================================
# 1. DATASET CONFIGURATION
# ============================================================

DATASET_PATH = Path(r"D:\CropGuardAI\data\plantvillage")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# 2. COLLECT IMAGE PATHS AND LABELS
# ============================================================

image_paths = []
labels = []

for class_dir in sorted(DATASET_PATH.iterdir()):

    if class_dir.is_dir():

        for image_path in class_dir.glob("*.*"):

            image_paths.append(str(image_path))
            labels.append(class_dir.name)


print("Total images:", len(image_paths))
print("Total labels:", len(labels))


# ============================================================
# 3. CREATE CLASS MAPPING
# ============================================================

class_names = sorted(set(labels))

class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}


print("\nClass mapping:")

for index, class_name in enumerate(class_names):
    print(index, "->", class_name)


# ============================================================
# 4. CREATE 70 / 15 / 15 STRATIFIED SPLIT
# ============================================================

# First split:
# 70% training
# 30% temporary

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)


# Second split:
# 15% validation
# 15% test

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels
)


print("\nDataset split:")
print("Training:", len(train_paths))
print("Validation:", len(val_paths))
print("Test:", len(test_paths))


# ============================================================
# 5. CONVERT CLASS NAMES TO INTEGER LABELS
# ============================================================

train_labels_int = [
    class_to_index[label]
    for label in train_labels
]

val_labels_int = [
    class_to_index[label]
    for label in val_labels
]

test_labels_int = [
    class_to_index[label]
    for label in test_labels
]


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


# ============================================================
# 7. IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):

    # Read image file
    image = tf.io.read_file(path)

    # Decode JPG / PNG / other supported image formats
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    # Resize to ResNet50 input size
    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    # Convert pixel values to float32
    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# 8. CREATE TF.DATA DATASET
# ============================================================

def create_dataset(paths, labels, training=False):

    # Convert Python lists into TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    # Load and resize images
    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Apply augmentation ONLY during training
    if training:

        dataset = dataset.map(
            lambda images, labels: (
                augmentation(images, training=True),
                labels
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Group images into batches
    dataset = dataset.batch(
        BATCH_SIZE
    )

    # Prepare future batches while the model is training
    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# 9. CREATE TRAIN / VALIDATION / TEST DATASETS
# ============================================================

train_ds = create_dataset(
    train_paths,
    train_labels_int,
    training=True
)

val_ds = create_dataset(
    val_paths,
    val_labels_int,
    training=False
)

test_ds = create_dataset(
    test_paths,
    test_labels_int,
    training=False
)


# ============================================================
# 10. VERIFY DATASETS
# ============================================================

print("\nTensorFlow datasets created!")

print("Training batches:", len(train_ds))
print("Validation batches:", len(val_ds))
print("Test batches:", len(test_ds))


# ============================================================
# 11. CHECK ONE TRAINING BATCH
# ============================================================

for images, labels_batch in train_ds.take(1):

    print("\nOne training batch:")

    print("Images shape:", images.shape)

    print("Labels shape:", labels_batch.shape)

    print("First 10 labels:", labels_batch[:10].numpy())