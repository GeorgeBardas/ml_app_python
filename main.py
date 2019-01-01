# Import TensoFlow
import tensorflow as tf
from tensorflow import keras
import cv2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# Load MINST Fashion Data
fashion_dataset = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_dataset.load_data()

# Class Labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# Build Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile Model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(train_images, train_labels, epochs = 15)

# Test accuracy
#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print(test_acc)

# Prediction
# predictions = model.predict(test_images)
#plt.figure()
#plt.imshow(test_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.xlabel(class_names[np.argmax(predictions[0])])
#plt.show()

image = cv2.imread('bag2.jpg', 0)
image = cv2.resize(image, (28, 28))
image_float = image.astype(float)
image_float = np.array(image_float) 
image_float = image_float / 255.0
image_float[image_float == 1] = 0
image_float = (np.expand_dims(image_float, 0))
print(image_float)

# Single prediction
predictions_single = model.predict(image_float)
plt.figure()
plt.imshow(image)
plt.xlabel(class_names[np.argmax(predictions_single[0])])
plt.show()

# Save tf.keras model in HDF5 format.
# keras_file = "keras_model.h5"
# tf.keras.models.save_model(model, keras_file)

# Convert to TensorFlow Lite model.
# converter = tf.converter.lite.TFLiteConverter.from_keras_model_file(keras_file)
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)