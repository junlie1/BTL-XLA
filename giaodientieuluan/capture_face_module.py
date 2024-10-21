import cv2
import os
import mysql.connector
from api import upload_image_to_firebase  # Hàm từ file api.py để tải ảnh lên Firebase
from datetime import datetime

def save_student_to_db(mssv, name, student_class):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="nhom3_xla"
    )
    mycursor = mydb.cursor()

    # Kiểm tra xem MSSV đã tồn tại chưa
    mycursor.execute("SELECT * FROM students WHERE mssv = %s", (mssv,))
    result = mycursor.fetchone()

    if result:
        print(f"Student with MSSV {mssv} already exists.")
    else:
        sql = "INSERT INTO students (mssv, name, student_class) VALUES (%s, %s, %s)"
        val = (mssv, name, student_class)
        mycursor.execute(sql, val)
        mydb.commit()
        print(f"Student {name} added successfully.")

def capture_faces(mssv, name, student_class):
    # Lưu sinh viên vào database trước
    save_student_to_db(mssv, name, student_class)

    # Sử dụng camera để chụp ảnh
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    print("Start capturing face images. Look at the camera...")
    count = 0
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            # Lưu ảnh tạm
            img_name = f"temp/User.{mssv}.{count}.jpg"
            cv2.imwrite(img_name, gray[y:y+h, x:x+w])

            # Tải ảnh lên Firebase
            firebase_url = upload_image_to_firebase(img_name)

            # Lưu đường dẫn ảnh vào MySQL
            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="nhom3_xla"
            )
            mycursor = mydb.cursor()
            sql = "INSERT INTO images (image_student, image_url) VALUES (%s, %s)"
            val = (mssv, firebase_url)
            mycursor.execute(sql, val)
            mydb.commit()

            print(f"Image {count} captured and uploaded to Firebase.")

        if count >= 30:  # Chụp 30 ảnh khuôn mặt
            break

    cam.release()
    cv2.destroyAllWindows()
