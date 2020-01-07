import tensorflow as tf
from tensorflow import keras
import numpy as np


data = keras.datasets.fashion_mnist
(trainEx, trainLbl), (testEx, testLbl) = data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

trainEx = trainEx/255.0
testEx = testEx/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(trainEx, trainLbl, epochs=5)
test_loss, test_acc = model.evaluate(testEx, testLbl)
prediction = model.predict(np.array(testEx))
for i in range(len(prediction)):
    print(class_names[np.argmax(prediction[i])])
print("Test Acc: ", test_acc)

