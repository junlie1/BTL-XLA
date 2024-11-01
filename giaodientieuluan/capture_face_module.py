import cv2
import os
from pymongo import MongoClient
from api import upload_image_to_firebase
from datetime import datetime

if not os.path.exists('temp'):
    os.makedirs('temp')

client = MongoClient('mongodb://localhost:27017/')
db = client['nhom2_xla']  # Tên database của bạn
students_collection = db['students']
images_collection = db['images']

def save_student_to_db(mssv, name, student_class):
    student = {
        "mssv": mssv,
        "name": name,
        "student_class": student_class
    }
    # Kiểm tra nếu sinh viên đã tồn tại
    if students_collection.find_one({"mssv": mssv}) is None:
        students_collection.insert_one(student)

def save_image_to_db(mssv, image_link):
    image_data = {
        "image_student": mssv,
        "image": image_link
    }
    images_collection.insert_one(image_data)

def capture_faces(mssv, name, student_class):
    save_student_to_db(mssv, name, student_class)
    vid_cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0

    while count < 30:
        _, image_frame = vid_cam.read()
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_image = gray[y:y+h, x:x+w]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_filename = f"User_{mssv}_{timestamp}.jpg"
            image_path = f"temp/{image_filename}"
            cv2.imwrite(image_path, face_image)

            image_link = upload_image_to_firebase(image_path, f"images/{image_filename}")
            save_image_to_db(mssv, image_link)
            cv2.imshow('frame', image_frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    vid_cam.release()
    cv2.destroyAllWindows()

    for file in os.listdir("temp"):
        os.remove(os.path.join("temp", file))

    return f"Captured and processed {count} face(s) for student {name} (MSSV: {mssv})."
