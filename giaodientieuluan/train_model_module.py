import cv2
import numpy as np
import mysql.connector
import requests
from PIL import Image
import io
import os

def get_student_images_from_db():
    """
    Truy vấn bảng `images` trong MySQL để lấy tất cả URL ảnh từ Firebase cho việc huấn luyện.
    """
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="nhom3_xla"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT image_student, image FROM images")
    images_data = mycursor.fetchall()
    mydb.close()
    return images_data

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    face_samples = []
    ids = []

    # Lấy ảnh từ Firebase qua URL và xử lý
    images_data = get_student_images_from_db()
    for mssv, image_url in images_data:
        try:
            # Tải ảnh từ URL Firebase
            response = requests.get(image_url)
            img = Image.open(io.BytesIO(response.content)).convert('L')
            img_numpy = np.array(img, 'uint8')

            # Phát hiện khuôn mặt
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(int(mssv))  # MSSV của sinh viên

        except Exception as e:
            print(f"Could not load image for student {mssv}: {e}")

    # Huấn luyện mô hình
    if len(face_samples) > 0:
        recognizer.train(face_samples, np.array(ids))
        if not os.path.exists('trainer'):
            os.makedirs('trainer')
        recognizer.save('trainer/trainer.yml')
        print("Training completed with LBPH.")
    else:
        print("No face samples found for training.")

    return "Training completed with LBPH."

