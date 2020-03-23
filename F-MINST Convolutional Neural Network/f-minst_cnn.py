import tensorflow as tf
from tensorflow import keras
import numpy as np


data = keras.datasets.fashion_mnist
(trainEx, trainLbl), (testEx, testLbl) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255
test_images = test_images / 255
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(trainEx, trainLbl, epochs=20)
test_loss, test_acc = model.evaluate(testEx, testLbl)
prediction = model.predict(np.array(testEx))
for i in range(len(prediction)):
    print(class_names[np.argmax(prediction[i])])
print("Test Acc: ", test_acc)
