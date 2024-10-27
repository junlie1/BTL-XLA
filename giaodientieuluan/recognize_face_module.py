import cv2
import mysql.connector

def get_student_info(mssv):
    """
    Truy vấn MySQL để lấy tên và lớp học của sinh viên dựa trên MSSV.
    """
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

def recognize_face():
    """
    Mở camera và nhận diện khuôn mặt trong thời gian thực bằng mô hình LBPH đã được huấn luyện.
    """
    # Khởi tạo bộ nhận diện khuôn mặt LBPH và tải mô hình đã huấn luyện
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')  # Đảm bảo bạn đã có file 'trainer.yml' từ quá trình huấn luyện

    # Khởi tạo bộ phát hiện khuôn mặt
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Mở camera
    video_capture = cv2.VideoCapture(0)

    while True:
        # Đọc từng khung hình từ camera
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyển khung hình sang grayscale
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)  # Phát hiện khuôn mặt

        for (x, y, w, h) in faces:
            # Nhận diện khuôn mặt dựa trên mô hình LBPH
            mssv, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # Nếu độ tin cậy dưới 100 thì nhận diện thành công
            if confidence < 100:
                student_info = get_student_info(mssv)
                if student_info:
                    name, student_class = student_info
                    confidence_text = f"{100 - confidence:.2f}%"
                    display_text = f"Name: {name}, MSSV: {mssv}, Confidence: {confidence_text}"
                else:
                    display_text = "Unknown person"
            else:
                display_text = "Unknown person"

            # Hiển thị thông tin nhận diện trên khung hình
            cv2.putText(frame, display_text, (x, y - 10), font, 0.75, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Hiển thị khung hình
        cv2.imshow('Face Recognition', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    video_capture.release()
    cv2.destroyAllWindows()

