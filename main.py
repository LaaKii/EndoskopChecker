import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np

data_path = "D:/dev/AI/pngs_labeled"

image_size = (396, 396)
butch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    labels='inferred',
    color_mode="grayscale",
    batch_size=butch_size,
    image_size=image_size,
    shuffle=True,
    seed=123
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    labels='inferred',
    color_mode="grayscale",
    batch_size=butch_size,
    image_size=image_size,
    shuffle=True,
    seed=123
)
class_names = train_ds.class_names



# plt.figure(figsize=(10,10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3,3,i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
#
# plt.show()


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(image_size[0],
                                                              image_size[1],
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(image_size[0], image_size[1], 1)),
    layers.Conv2D(16, 1, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 1, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 1, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(92, 1, activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(2)
])

model.summary()

model.compile(
    optimizer = "adam",
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
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

img = tf.keras.preprocessing.image.load_img(
    "D:/dev/AI/pngs_labeled/213_Bildhomogenitaet_LED_Leuchfeld_3_894.png", target_size=None, color_mode="grayscale"
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

prediction = model.predict(img_array)
score = tf.nn.softmax(prediction[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)