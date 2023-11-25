#!/usr/bin/env python
# coding: utf-8

from io import BytesIO
from urllib import request

from PIL import Image

import numpy as np

import tflite_runtime.interpreter as tflite


MODEL_PATH = 'bees-wasps-v2.tflite'
IMAGE_TARGET_SIZE = (150, 150)


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def convert_image_to_input(img):
    x = np.array(img, dtype='float32')
    X = np.array([x]) / 255.0
    return X


def setup_tflite_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    return interpreter, input_index, output_index


def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=IMAGE_TARGET_SIZE)
    X = convert_image_to_input(img)
    interpreter, input_index, output_index = setup_tflite_interpreter(MODEL_PATH)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    return pred[0].tolist()


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
