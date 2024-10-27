import base64

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from capture_face_module import capture_faces
from train_model_module import train_model
from recognize_face_module import recognize_face

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dataset', methods=['POST'])
def dataset():
    data = request.get_json()
    mssv = data.get('mssv')
    name = data.get('name')
    student_class = data.get('student_class')

    # Gọi hàm capture_faces và truyền các thông tin sinh viên
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


if __name__ == '__main__':
    app.run(debug=True)
