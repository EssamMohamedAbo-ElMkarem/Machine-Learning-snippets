import cv2
import os
import numpy as np
from PIL import Image
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")
face_cascade = cv2.CascadeClassifier("C:\Python\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")
# Recognizer training
recognizer = cv2.face.LBPHFaceRecognizer_create()
current_id = 0
label_ids = {}
y_labels = list()
x_train = list()

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-")
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            print(label_ids)
            pil_image = Image.open(path).convert("L")  # Gray scale
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for(x, y, h, w) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
            print(image_array)

print(y_labels)
print(x_train)
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
