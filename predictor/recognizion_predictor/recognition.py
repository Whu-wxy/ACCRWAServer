#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 12:43:36 2020

# @author: chen
# """
import numpy as np
import random
import tensorflow as tf
import cv2
from nets import nets_factory
import json
from tensorflow.python.training import saver as tf_saver
slim = tf.contrib.slim

def cv_imread(file_path):
    img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    img = img.astype('uint8')
    cv_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return cv_img

def cv_preprocess_image(img, output_height, output_width):
    assert output_height == output_width
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:, :, 0] = np.uint8((np.int32(img[:, :, 0]) + (180 + random.randrange(-9, 10))) % 180)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    rows, cols, ch = img.shape
    output_size = output_width
    def r():
        return (random.random() - 0.5) * 0.1 * output_size
    pts1 = np.float32([[0, 0], [cols, rows], [0, rows]])
    pts2 = np.float32([[r(), r()], [output_size + r(), output_size + r()], [r(), output_size + r()]])
    M = cv2.getAffineTransform(pts1, pts2)
    noize = np.random.normal(0, random.random() * (0.05 * 255), size=img.shape)
    img = np.array(img, dtype=np.float32) + noize
    img = cv2.warpAffine(img, M, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img

def tf_preprocess_image(image, output_height, output_width):
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [output_height, output_width],                                             align_corners=False)
    image = tf.squeeze(image, axis=0)
    image = tf.to_float(image)
    return image

def get_tf_preprocess_image():

    def foo(image, output_height, output_width):
        image = tf_preprocess_image(
        image, output_height, output_width)
        return tf.image.per_image_standardization(image)
    return foo

# tf.reset_default_graph()

n = 1

img = cv_imread('../../0.jpg')
image = cv_preprocess_image(img, 224,224)
image2 = np.expand_dims(image, 0)

image_preprocessing_fn = get_tf_preprocess_image()
images_holder = [tf.placeholder(tf.uint8, shape=(None, None, 3)) for i in range(n)]
network_fn = nets_factory.get_network_fn('resnet_v2_50', 3755, weight_decay=0.0, is_training=False)
eval_image_size = 224
images = [image_preprocessing_fn(images_holder[i], eval_image_size, eval_image_size) for i in range(n)]
logits, _ = network_fn(images)
eval_ops = logits
variables_to_restore = slim.get_variables_to_restore()

def load_model():
    #tf.reset_default_graph()
    checkpoint_path = './products/train_logs_resnet_v2_50/model.ckpt-100000'
    with tf.Session() as session:
        saver = tf_saver.Saver(variables_to_restore)
        saver.restore(session, checkpoint_path)
        results = []
        lo = 0
        while lo != len(image2):
            hi = min(len(image2), lo + n)
            feed_data = image2[lo:hi]
            logits = session.run(eval_ops, feed_dict={images_holder[i]: feed_data[i] for i in range(n)})
            results.append(logits[:hi - lo])
            logits = np.concatenate(results, axis=0)
            lo = hi

    return logits

def main():
    with open('./products/cates.json') as f:
        lines = f.read()
        cates = json.loads(lines.strip())
    logits = load_model()
    assert 3755  == logits.shape[1]
    logits = logits[:, :3755]
    explogits = np.exp(np.minimum(logits, 70))
    expsums = np.sum(explogits, axis=1)
    expsums.shape = (logits.shape[0], 1)
    expsums = np.repeat(expsums,3755, axis=1)
    probs = explogits / expsums
    argsorts = np.argsort(-logits, axis=1)
    lo = 0
    # for i in range(n):
    m = 1
    predictions = []
    probabilities = []
    for i in range(m):
        pred = argsorts[lo][:5]
        prob = probs[lo][pred].tolist()
        pred = list(map(lambda i: cates[i]['text'], pred.tolist()))
        predictions.append(pred)
        probabilities.append(prob)
        lo += 1
    for i in range (5):
        print('predictions',predictions[0][i], probabilities[0][i])


if __name__ == '__main__':
    main()

