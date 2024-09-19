import cv2
import numpy as np
import json
import cv2
from PIL import Image
import os


class StudentFaceManager:

    def __init__(self):
        pass

    def create_directory(self, directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_face_id(self, directory: str) -> int:
        user_ids = []
        for filename in os.listdir(directory):
            number = int(os.path.split(filename)[-1].split("-")[1])
            user_ids.append(number)
        user_ids = sorted(list(set(user_ids)))
        max_user_ids = 1 if len(user_ids) == 0 else max(user_ids) + 1
        for i in sorted(range(0, max_user_ids)):
            try:
                if user_ids.index(i):
                    # continue
                    face_id = 1
            except ValueError as e:
                return i
        return max_user_ids

    def save_name(self, face_id: int, face_name: str, filename: str) -> None:
        names_json = None
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                names_json = json.load(fs)
        if names_json is None:
            names_json = {}
        names_json[face_id] = face_name
        with open(filename, 'w') as fs:
            json_dump = json.dumps(names_json, ensure_ascii=False, indent=4)
            fs.write(json_dump)

    def start(self) -> None:
        if __name__ == '__main__':
            directory = 'images'
            cascade_classifier_filename = 'haarcascade_frontalface_default.xml'
            names_json_filename = 'names.json'

            self.create_directory(directory)
            faceCascade = cv2.CascadeClassifier(cascade_classifier_filename)
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)
            cam.set(4, 480)
            count = 0
            face_name = input('\nEnter user name:  ')
            face_id = self.get_face_id(directory)
            self.save_name(face_id, face_name, names_json_filename)
            print(
                '\nInitializing face capture. Look at the camera and wait...')

            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # minNeighbors=5 higher value reduce false positives
                faces = faceCascade.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    count += 1
                    cv2.imwrite(
                        f'./images/Users-{face_id}-{count}.jpg', gray[y:y+h, x:x+w])
                    cv2.imshow('image', img)

                k = cv2.waitKey(100) & 0xff
                if k < 30:
                    break

                elif count >= 30:
                    break

            print('\nSuccess! Exiting Program.')
            cam.release()
            cv2.destroyAllWindows()


class StudentFaceTrainer:

    def __init__(self):
        pass

    if __name__ == "__main__":
        path = './images/'
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        def getImagesAndLabels(self, path):
            print("\nTraining...")
            detector = cv2.CascadeClassifier(
                "haarcascade_frontalface_default.xml")
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                id = int(os.path.split(imagePath)[-1].split("-")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    # Extract face region and append to the samples
                    faceSamples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(id)

            return faceSamples, ids

        def start(self):
            faces, ids = self.getImagesAndLabels(self.path)
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.write('trainer.yml')
            print("\n[INFO] {0} faces trained. Exiting Program".format(
                len(np.unique(ids))))


class StudentFaceRecognizer:
    if __name__ == "__main__":
        def start(self):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('trainer.yml')
            # print(recognizer)
            face_cascade_Path = "haarcascade_frontalface_default.xml"
            faceCascade = cv2.CascadeClassifier(face_cascade_Path)
            font = cv2.FONT_HERSHEY_SIMPLEX
            id = 0
            names = ['None']
            with open('names.json', 'r') as fs:
                names = json.load(fs)
                names = list(names.values())
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)
            cam.set(4, 480)
            minW = 0.1 * cam.get(3)
            minH = 0.1 * cam.get(4)

            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)),)
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    id, confidence = recognizer.predict(
                        gray[y:y + h, x:x + w])
                    if confidence > 51:
                        try:
                            name = names[id]
                            confidence = "  {0}%".format(round(confidence))
                        except IndexError as e:
                            name = "Who are you?"
                            confidence = "N/A"
                    else:
                        name = "Who are you?"
                        confidence = "N/A"
                    cv2.putText(img, name, (x + 5, y - 5),
                                font, 1, (255, 255, 255), 2)
                    cv2.putText(img, confidence, (x + 5, y + h - 5),
                                font, 1, (255, 255, 0), 1)
                cv2.imshow('camera', img)
                k = cv2.waitKey(10) & 0xff
                if k == 27:
                    break
            print("\n [INFO] Exiting Program.")
            cam.release()
            cv2.destroyAllWindows()


exit = False
while (not exit):

    print("**************** face Recognizer ****************")
    print("1- to take a take pictures")
    print("2- train the program on the dataset")
    print("3-luanch face recognizer")
    choice = input("select an option: ")
    if choice == "1":
        studentFaceManager = StudentFaceManager()
        studentFaceManager.start()
    elif choice == "2":
        studentFaceTrainer = StudentFaceTrainer()
        studentFaceTrainer.start()
    elif choice == "3":
        studentFaceRecognizer = StudentFaceRecognizer()
        studentFaceRecognizer.start()
    else:
        print("Invalid choice. Please select 1, 2, or 3.")
        exit = True
