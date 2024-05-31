from flask import Flask, request, jsonify, render_template
import os
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        result = subprocess.run(['python3', 'predict_disease.py', file_path], capture_output=True, text=True)
        predictions = result.stdout.strip()

        # Handle the case where predictions might be empty
        if predictions == "":
            predictions = "Healthy"

        return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
