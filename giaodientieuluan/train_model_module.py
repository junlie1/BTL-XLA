import cv2
import numpy as np
import os
import mysql.connector

def get_images_and_labels():
    # Kết nối tới MySQL để lấy các đường dẫn ảnh từ bảng 'images'
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="nhom3_xla"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT image_student, image_url FROM images")
    result = mycursor.fetchall()

    face_samples = []
    ids = []

    for (student_id, image_url) in result:
        # Đọc từng ảnh từ đường dẫn đã lưu trong Firebase
        img_numpy = np.array(cv2.imread(image_url), 'uint8')
        faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(student_id)

    return face_samples, ids

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, ids = get_images_and_labels()

    # Huấn luyện mô hình với các khuôn mặt và MSSV tương ứng
    recognizer.train(faces, np.array(ids))

    # Lưu mô hình đã huấn luyện
    recognizer.write('trainer/trainer.yml')
    print("Model training completed and saved to trainer.yml")
