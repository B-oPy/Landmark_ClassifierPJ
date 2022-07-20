from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './MODEL/010-0.1822-0.9444-0.2505-0.9222.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        landmark = ['burj_khalifa','chichen_itza','christ_the_reedemer','eiffel_tower','great_wall_of_china',
                    'machu_pichu','pyramids_of_giza','roman_colosseum','statue_of_liberty','stonehenge',
                    'taj_mahal','venezuela_angel_falls']
        classes = model_predict(file_path, model)
        import operator

        result ={}
        for i in range(12) :
            result[landmark[i]] = round(classes[0][i]*100,4)
        r1 = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
        result= str(r1[0][0]) + ' : ' + str(r1[0][1]) + '%\n' + str(r1[1][0]) + ' : ' + str(r1[1][1]) + '%\n' + str(r1[2][0]) + ' : ' + str(r1[2][1]) + '%'

        return result
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

