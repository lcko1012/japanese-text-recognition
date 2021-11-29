import os
import time
import cv2
import numpy
from flask import Flask, request, url_for, render_template, flash, jsonify, make_response, send_file
from flask_cors import CORS, cross_origin
from werkzeug.utils import redirect
from detector import detector
from demo import predict
import base64
from pathlib import Path
import random, string


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = '/var/www/html/input/'
app.config['CORS_HEADERS']='Context-Type'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


@app.route('/')
@cross_origin()
def home():
    return jsonify(
        message="BACKEND SERVER - JAPANESE RECOGNITION"
    )


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_folder():
    root_output_dir = Path('/var/www/html/uploads/')
    # set up logger
    if not root_output_dir.exists():
        root_output_dir.mkdir()
        print('create')

    letters = string.ascii_lowercase
    randomly = ''.join(random.choice(letters) for i in range(5))
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S-') + randomly
    images_output_dir = root_output_dir / time_str / 'input'
    # images_output_dir.mkdir(parents=True, exist_ok=True)
    detector_outputs_dir = root_output_dir / time_str / 'detector_outputs'
    detector_outputs_dir.mkdir(parents=True, exist_ok=True)
    return {'image_dir': str(images_output_dir), 'detect_dir': str(detector_outputs_dir)}


@app.route('/predict', methods=['POST'])
@cross_origin()
def upload_file():
    uploads = create_folder()
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        npimg = numpy.fromstring(file.read(), numpy.uint8)  # convert string data to numpy array
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)  # convert numpy array to image
        # path = os.path.join(app.config['UPLOAD_FOLDER']) + "image.png"
        start = time.time()
        # cv2.imwrite(path, img)
        try:
            result = []
            # array image folder
            image_folder = detector(img, uploads['detect_dir'])
            for i in range(len(image_folder)):
                print(image_folder[i])
                result.append(predict(image_folder[i]))
            end = time.time()
            image_text_detection = uploads['detect_dir'] + "/image_text_detection.png"
            with open(image_text_detection, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                file = encoded_string
            return jsonify(
                result=result,
                time=end - start,
                img=file.decode()
            )
        except Exception as e:
            print(e)
            return make_response(jsonify(message='An error has occurred'), 400)
    else:
        return make_response(jsonify(message="Allowed image types are png, jpg, jpeg"), 400)

    return jsonify(
        message="hello"
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0')