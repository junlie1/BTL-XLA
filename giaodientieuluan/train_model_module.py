import cv2
import numpy as np
from pymongo import MongoClient
import requests
from PIL import Image
import io
import os

client = MongoClient('mongodb://localhost:27017/')
db = client['nhom2_xla']
students_collection = db['students']
images_collection = db['images']

def get_student_images_from_db():
    images_data = images_collection.find({}, {"image_student": 1, "image": 1})
    return [(img['image_student'], img['image']) for img in images_data]

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    face_samples = []
    ids = []

    images_data = get_student_images_from_db()
    for mssv, image_url in images_data:
        try:
            response = requests.get(image_url)
            img = Image.open(io.BytesIO(response.content)).convert('L')
            img_numpy = np.array(img, 'uint8')

            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(int(mssv))
                print(f"Training on face for student ID {mssv}")

        except Exception as e:
            print(f"Could not load image for student {mssv}: {e}")

    if len(face_samples) > 0:
        recognizer.train(face_samples, np.array(ids,dtype=np.int32))
        if not os.path.exists('trainer'):
            os.makedirs('trainer')
        recognizer.save('trainer/trainer.yml')
        print("Training completed with LBPH.")
    else:
        print("No face samples found for training.")

    return "Training completed with LBPH."

