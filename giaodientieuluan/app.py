import base64
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from capture_face_module import capture_faces
from train_model_module import train_model
from recognize_face_module import capture_and_attend, recognize_face

app = Flask(__name__)

def extract_attendance():
    """Đọc dữ liệu từ file CSV và trả về dưới dạng danh sách các từ điển."""
    try:
        today = datetime.now().strftime("%m_%d_%y")
        df = pd.read_csv(f'Attendance/Attendance-{today}.csv')
        return df.to_dict(orient='records')  # Chuyển DataFrame thành danh sách các từ điển
    except FileNotFoundError:
        return []  # Nếu file không tồn tại, trả về danh sách rỗng

@app.route('/')
def index():
    attendance_data = extract_attendance()
    return render_template('index.html', attendance_data=attendance_data)

@app.route('/dataset', methods=['POST'])
def dataset():
    data = request.get_json()
    mssv = data.get('mssv')
    name = data.get('name')
    student_class = data.get('student_class')

    result = capture_faces(mssv, name, student_class)
    return jsonify({"message": result})


@app.route('/train', methods=['POST'])
def train():
    result = train_model()
    return jsonify({"message": result})


@app.route('/recognize', methods=['GET'])
def recognize():
    try:
        # Gọi hàm recognize_face_from_camera để mở camera và nhận diện
        recognize_face()
        return jsonify({"message": "Recognition completed"})
    except Exception as e:
        print(f"Error during recognition: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/attendance', methods=['POST'])
def attendance():
    name, mssv, current_time = capture_and_attend()
    if name and mssv:
        current_time = datetime.now().strftime("%H:%M:%S")
        return jsonify({"success": True, "name": name, "mssv": mssv, "time": current_time})
    else:
        return jsonify({"success": False, "message": "No face recognized or failed to capture image."})
if __name__ == '__main__':
    app.run(debug=True)