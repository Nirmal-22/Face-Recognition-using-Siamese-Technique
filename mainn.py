from __future__ import absolute_import
from __future__ import print_function
from layers import L2Dist
import os
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from keras import backend as K
import tensorflow as tf
from flask import Flask, request, jsonify
import glob
import pandas as pd
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)


haar_cascade1 = cv.CascadeClassifier('haar_face.xml')
haar_cascade2 = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')


def preprocess(path):
    img = cv.imread(path, 0)
    faces_rect = haar_cascade1.detectMultiScale(img, 1.3, 5)
    if (len(faces_rect) == 0):
        faces_rect = haar_cascade2.detectMultiScale(img, 1.3, 5)
    if (len(faces_rect) == 0):
        return np.array([1])
    for (x, y, w, h) in faces_rect:
        img = img[x:x+w+10, y:y+h]
        break
    img = cv.resize(img, (47, 62))
    img = np.expand_dims(img, -1)
    img = img/255.0
    return img


def generate_test_image_pairs(images_dataset, image):
    pair_images = []
    for img in images_dataset:
        pair_images.append([image, img])
    return np.array(pair_images)


def predict_found(model, image_pair, thres, names):
    temp = model.predict([image_pair[:, 0], image_pair[:, 1]])
    ans = ""
    if sum(temp < thres)[0]:
        # print(temp)
        idx = np.argmin(temp)
        # print(idx)
        ans = "Welcome "+names[idx]
    else:
        ans = "Not found"
    return ans


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


siamese_model = tf.keras.models.load_model('siamesemodelv2.h5',
                                           custom_objects={'L2Dist': L2Dist, 'contrastive_loss': contrastive_loss})

thres = 0.45

data_path = 'images/samples/'

test_path = 'images/test/'

# print(predict_found(siamese_model,pair,thres,names))


@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():

    if request.method == 'POST':
        filelist = glob.glob(os.path.join(test_path, "*"))
        for f in filelist:
            os.remove(f)

        # print("hi")
        file = request.files['image']
        # print('hello')
        file.save('images/test/image.jpg')

    img_test = os.listdir(test_path)[0]
    test_img = preprocess(test_path+img_test)
    if test_img.shape != (1,):
        images = []

        df = pd.read_csv('names.csv')
        names = df['name'].tolist()

        for i, row in df.iterrows():
            # print(row['img'])
            img = preprocess(row['img'])
            images.append(img)

        pair = generate_test_image_pairs(images, test_img)
        # Read the image via file.stream
        # img = Image.open(file.stream)

        response = predict_found(siamese_model, pair, thres, names)
    else:
        response = "No face detected, Try Again"

    return jsonify(response)


@app.route("/register", methods=["GET", "POST"])
def register_user():

    if request.method == "POST":
        try:
            # print("hi")
            file = request.files['image']
            user_name = request.form['name']
            # print('hello')

            file_path = data_path + user_name + '.jpg'
            file.save(file_path)

            df = pd.read_csv('names.csv', index_col=0)
            #print(len(df.index), user_name, file_path)
            df.loc[len(df.index)] = [user_name, file_path]
            df.to_csv('names.csv')
            return "Registration successfully"

        except Exception as e:
            return "error"


if __name__ == "__main__":
    app.run(host='0.0.0.0')
