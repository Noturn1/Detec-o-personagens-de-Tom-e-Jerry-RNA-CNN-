import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

IMAGE_DIRECTORY = ".\data"
batch_size = 32
img_height = 72
img_width = 128

train_ds = tf.keras.utils.image_dataset_from_directory(
    IMAGE_DIRECTORY,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    IMAGE_DIRECTORY,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

plt.figure(figsize=(15,5))
for i, (images, labels) in enumerate(train_ds.take(1)):
    for j in range(10):
        ax = plt.subplot(2, 5, j + 1)
        plt.imshow(images[j].numpy().astype("uint8"))
        plt.title(f"Label: {labels[j].numpy()}")
        plt.axis("off")
plt.show()

# Number of classes
num_classes = len(set(label for _, labels in train_ds for label in labels.numpy()))
print("Number of Classes:", num_classes)

label_categories = {
    0: "Jerry",
    1: "Tom",
    2: "None",
    3: "Both"
}

#CNN model arquitecture
# CNN model architecture
cnn_model = Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

cnn_model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

# Traininig the model
epochs = 10
history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
cnn_model.summary()
cnn_model.save("models/cnn_model2.h5")

# Plotting the training history
# accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Predicting one image from the validation dataset
plt.figure(figsize=(6, 3))  
for images, labels in val_ds.take(1):
    sample_image = images[0]
    true_label = labels[0]

    sample_image = tf.expand_dims(sample_image, axis=0)

    predictions = cnn_model.predict(sample_image)

    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_class = label_categories[predicted_class_index]

    plt.imshow(sample_image[0].numpy().astype("uint8"))
    plt.title(f"True label: {label_categories[true_label.numpy()]}\nPredicted label: {predicted_class}")
    plt.axis('off')

plt.show()

