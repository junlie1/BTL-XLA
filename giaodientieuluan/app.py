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
    mssv = request.form['mssv']
    name = request.form['name']
    student_class = request.form['class']
    result = capture_faces(mssv, name, student_class)
    return jsonify({"message": result})

@app.route('/train', methods=['POST'])
def train():
    result = train_model()
    return jsonify({"message": result})

@app.route('/recognize', methods=['POST'])
def recognize():
    result = recognize_face()
    return jsonify({"message": result})

if __name__ == '__main__':
    app.run(debug=True)
