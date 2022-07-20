from __future__ import division, print_function
# from crypt import methods
# coding=utf-8
import sys



import os
import glob
import re
import numpy as np

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
MODEL_PATH = 'C:/Users/bnui0/Downloads/mobilenet-010-0.1991-0.9538.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# model.save('ResNet50_modle.h5')
print('Model loaded. Check http://127.0.0.1:5001/')


def model_predict(img_path, model):
    from keras.models import load_model
    from keras.preprocessing import image
    from keras.applications.imagenet_utils import preprocess_input, decode_predictions
    from PIL import Image, ImageOps
    import numpy as np
    import pandas as pd

    # Load the model
    model = load_model('C:/Users/bnui0/Downloads/mobilenet-010-0.1991-0.9538.hdf5')
    img_path = 'C:/Users/bnui0/Downloads/landmark_Flask_Bopy/landmark_Flask_Bopy/uploads/0dc7fe2967.jpg'
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    # run the inference
    prediction = model.predict(x)
    print(prediction)

    landmark = ['burj_khalifa','chichen_itza','christ_the_reedemer','eiffel_tower','great_wall_of_china','machu_pichu',
                'pyramids_of_giza','roman_colosseum','statue_of_liberty','stonehenge','taj_mahal','venezuela_angel_falls']
    landlist = pd.Series(landmark, name='landmark').astype(str)
    predlist = pd.Series(prediction[0]*100, name='prediction')

    # 경로와 라벨 concatenate
    df = pd.concat([landlist, predlist], axis=1, )
    df = df.sort_values(by=['prediction'],ascending=False,ignore_index=True).round(6)
    return df

# Model select page
@app.route('/', methods=['GET'])
def model_Select():
    return render_template('select.html')


@app.route('/mobilenet')
def select():
    # Main page
    return render_template('mobilenet.html')

@app.route('/vgg16')
def index():
    # Main page
    return render_template('vgg16.html')    

@app.route('/vgg19')
def index1():
    # Main page
    return render_template('vgg19.html')

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
        # preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        # return result
        landmark = ['burj_khalifa','chichen_itza','christ_the_reedemer','eiffel_tower','great_wall_of_china',
                    'machu_pichu','pyramids_of_giza','roman_colosseum','statue_of_liberty','stonehenge',
                    'taj_mahal','venezuela_angel_falls']
        classes = model_predict(file_path, model)
        # for i in range(len(classes[0])) :
        #     if classes[0][i]>40:
                # result = landmark[i]
        result = classes.head(5)
        return result
    return None

#classes = df

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)

