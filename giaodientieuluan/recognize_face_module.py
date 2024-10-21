import cv2
import mysql.connector
import numpy as np

def recognize_face():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 100:
                # Kết nối tới MySQL để lấy thông tin sinh viên dựa vào MSSV
                mydb = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="",
                    database="nhom3_xla"
                )
                mycursor = mydb.cursor()
                mycursor.execute("SELECT name, student_class FROM students WHERE mssv = %s", (id,))
                result = mycursor.fetchone()

                if result:
                    name, student_class = result
                    text = f"{name} - {student_class} (Confidence: {round(100 - confidence)}%)"
                else:
                    text = "Unknown"

            else:
                text = "Unknown"

            cv2.putText(frame, text, (x+5, y-5), font, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Camera', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
