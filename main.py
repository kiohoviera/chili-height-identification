import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
import json
import os
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cam = cv2.VideoCapture(0)

img_height = 224
img_width = 224

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset folder")

args = vars(ap.parse_args())
tf.keras.utils.disable_interactive_logging()

train_ds = tf.keras.utils.image_dataset_from_directory(
    args['dataset'],
    validation_split=0.2,
    subset='training',
    image_size=(img_height, img_width),
    batch_size=32,
    seed=42,
    shuffle=True)


def identify():
    model = tf.keras.models.load_model('chili.h5')

    detection_path = 'input_image.jpg'
    img = tf.keras.utils.load_img(
        detection_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])

    return train_ds.class_names[np.argmax(score)]


def height(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(contours)
    cm = h * 0.0264583333

    return cm


def identifyColor(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
    imask = mask > 0
    green = np.zeros_like(image, np.uint8)
    green[imask] = image[imask]

    objectHeight = height(mask)

    return objectHeight


def sendData(height, stage, image):
    url = "https://npk.aviarthardph.net/sendData"
    files = {'identified': open(image, 'rb')}
    rData = {'measurement': height, 'stage': stage}

    r = requests.post(url, files=files, data=rData)

    print(r.text)

def capture():

    result, image = cam.read()
    if result:

        cv2.imshow("SoilMonitoringCapture", image)

        cv2.imwrite("input_image.jpg", image)
    else:
        print("Unable to capture image, please check your camera")

if __name__ == '__main__':
    capture()
    image = cv2.imread('input_image.jpg')
    stage = identify()
    measurement = identifyColor(image)

    sendData(measurement, stage, 'input_image.jpg')
