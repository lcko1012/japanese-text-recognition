import os
import time

import cv2
import numpy
from flask import Flask, request, url_for, render_template, flash, jsonify, make_response, send_file
from flask_cors import CORS
from werkzeug.utils import redirect
from detector import detector
from demo import predict
import base64
import app as app

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'input/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


@app.route('/')
def home():
    return jsonify(
        message="hello world"
    )


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        npimg = numpy.fromstring(file.read(), numpy.uint8)  # convert string data to numpy array
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)  # convert numpy array to image
        path = os.path.join(app.config['UPLOAD_FOLDER']) + "image.png"
        start = time.time()
        cv2.imwrite(path, img)
        try:
            result = []
            # array image folder
            image_folder = detector(img)
            for i in range(len(image_folder)):
                print(image_folder[i])
                result.append(predict(image_folder[i]))
            end = time.time()
            image_text_detection = "detector_outputs/image_text_detection.png"
            with open(image_text_detection, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                file = encoded_string
            return jsonify(
                result=result,
                time=end - start,
                img=file.decode()
            )
        except:
            return make_response(jsonify(message='An error has occurred'), 400)
    else:
        return make_response(jsonify(message="Allowed image types are png, jpg, jpeg"), 400)

    return jsonify(
        message="hello"
    )


if __name__ == '__main__':
    app.run()
