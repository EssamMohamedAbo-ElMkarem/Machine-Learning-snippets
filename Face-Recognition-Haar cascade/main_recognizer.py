__author__ = "Essam Mohamed"


class FaceRecognizer(object):
    def __init__(self):

        cascade_string = "C:\Python\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_string)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(r"\trainer.yml")
        self.labels = {}
        with open(r"\labels.pickle", "rb") as f:
            brought_labels = pickle.load(f)
            self.labels = {value: key for key, value in brought_labels.items()}
        self.cap = cv2.VideoCapture(0)

        while True:
            # Capturing frame by frame
            self.ret, self.frame = self.cap.read()
            # Converting from RGB  to Gray scale
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            # Main face detection process
            self.faces = self.face_cascade.detectMultiScale(self.gray, scaleFactor=1.5, minNeighbors=5)
            # Drawing rectangle around faces
            for (x, y, w, h) in self.faces:
                self.roi_gray = self.gray[y:y+h, x:x+w]
                self.roi_color = self.frame[y:y+h, x:x+w]
                # Main Identification process

                self.id_, self.conf = self.recognizer.predict(self.roi_gray)

                if self.conf >= 45:
                    print(self.labels[self.id_])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(self.frame, self.labels[self.id_], (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

                self.make_rectangle(self.frame, x, y, w, h)
                print(x, y, w, h)

            cv2.imshow("PyFace", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def make_rectangle(frame, x, y, w, h):
        color = (255, 0, 0)  # Stands for Blue in RGB system
        stroke = 2
        end_x = x + w
        end_y = y + h
        cv2.rectangle(frame, (x, y), (end_x, end_y), color, stroke)


if __name__ == "__main__":
    import numpy as np
    import cv2
    import pickle
    app = FaceRecognizer()
