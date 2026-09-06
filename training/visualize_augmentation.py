import tensorflow as tf
import matplotlib.pyplot as plt

from prepare_dataset import train_ds, class_names


for images, labels in train_ds.take(1):

    plt.figure(figsize=(12, 12))

    for i in range(9):

        plt.subplot(3, 3, i + 1)

        plt.imshow(
            images[i].numpy().astype("uint8")
        )

        class_index = labels[i].numpy()

        plt.title(class_names[class_index])

        plt.axis("off")

    plt.tight_layout()
    plt.show()