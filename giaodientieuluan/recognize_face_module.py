import cv2
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
import pandas as pd

# Kết nối đến MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['nhom2_xla']
students_collection = db['students']


# Hàm lấy thông tin sinh viên dựa trên MSSV
def get_student_info(mssv):
    student = students_collection.find_one({"mssv": str(mssv)})
    if student:
        return student['name'], student['student_class']
    else:
        print(f"No student found with MSSV: {mssv}")
    return None


# Hàm lưu thông tin vào tệp điểm danh
def save_attendance(name, mssv):
    today = datetime.now().strftime("%m_%d_%y")
    filename = f'Attendance/Attendance-{today}.csv'
    current_time = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists('Attendance'):
        os.makedirs('Attendance')

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if mssv not in df['Roll'].values:
            new_entry = pd.DataFrame([[name, mssv, current_time]], columns=['Name', 'Roll', 'Time'])
            df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = pd.DataFrame([[name, mssv, current_time]], columns=['Name', 'Roll', 'Time'])

    df.to_csv(filename, index=False)
    print(f"Attendance for {name} (MSSV: {mssv}) recorded at {current_time}.")


# Hàm nhận diện khuôn mặt từ video trực tiếp và hiển thị thông tin
def recognize_face():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")

    font = cv2.FONT_HERSHEY_SIMPLEX
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Nhận diện khuôn mặt trong khung đã phát hiện
            face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            confidence_text = f"{100 - confidence:.2f}%"  # Độ tin cậy

            if confidence < 100:
                student_info = get_student_info(str(face_id))
                if student_info:
                    name, student_class = student_info
                    display_text = f"Name: {name}, MSSV: {face_id}, Class: {student_class}"
                else:
                    display_text = "Unknown person"
            else:
                display_text = "Unknown person"

            # Hiển thị thông tin sinh viên và độ tin cậy trên khung hình
            cv2.putText(frame, display_text, (x, y - 10), font, 0.75, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence_text}", (x, y + h + 20), font, 0.75, (255, 255, 255), 2)

            # Nhận diện mắt
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eyeCascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)  # Màu xanh cho mắt

            # Nhận diện miệng
            smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)  # Màu đỏ cho miệng

        cv2.imshow('Face Recognition', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# Hàm chụp ảnh và lưu vào bảng điểm danh khi nhấn "Điểm danh"
def capture_and_attend():
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    video_capture.release()

    if not ret:
        print("Could not capture image.")
        return None, None

    # Lưu ảnh chụp
    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, frame)

    # Nhận diện từ ảnh chụp
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if confidence < 100:
            student_info = get_student_info(face_id)
            if student_info:
                name, student_class = student_info
                save_attendance(name, face_id)
                return name, face_id

    return None, None
