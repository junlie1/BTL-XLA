import cv2
import mysql.connector
from datetime import datetime
import os
import pandas as pd

# Hàm lấy thông tin sinh viên dựa trên MSSV
def get_student_info(mssv):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="nhom3_xla"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT name, student_class FROM students WHERE mssv = %s", (mssv,))
    result = mycursor.fetchone()
    mydb.close()
    return result

# Hàm nhận diện khuôn mặt từ video trực tiếp và hiển thị thông tin
def recognize_face():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            mssv, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            confidence_text = f"{100 - confidence:.2f}%"  # Độ tin cậy

            if confidence < 100:
                student_info = get_student_info(mssv)
                if student_info:
                    name, student_class = student_info
                    display_text = f"Name: {name}, MSSV: {mssv}"
                else:
                    display_text = "Unknown person"
            else:
                display_text = "Unknown person"

            # Hiển thị tên và MSSV phía trên khung hình
            cv2.putText(frame, display_text, (x, y - 10), font, 0.75, (255, 255, 255), 2)
            # Vẽ hình chữ nhật quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Hiển thị độ tin cậy phía dưới hình chữ nhật
            cv2.putText(frame, f"Confidence: {confidence_text}", (x, y + h + 20), font, 0.75, (255, 255, 255), 2)

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
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        mssv, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 100:
            student_info = get_student_info(mssv)
            if student_info:
                name, student_class = student_info
                save_attendance(name, mssv)
                return name, mssv

    return None, None

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
