# coding=UTF-8<code>
import gevent
from gevent import monkey   #多线程
monkey.patch_all()

from typing import List, Callable, TypeVar
import argparse
import json
import logging
import os
import sys
from flask import Flask, request, Response, jsonify, url_for
from flask_cors import CORS
import time
import numpy as np

from gevent.pywsgi import WSGIServer
from predictor.Predictor import Predictor
from predictor.recognizion_predictor.recognize_predictor import Recognize_Predictor_batch
from utils import check_for_gpu, allowed_file, draw_bbox, cvImg_to_base64, load_boxes, cv2_to_base64
from celery import Celery
from celery.app.task import Task
from celery.app.control import Control
from Sqlite3.sqlite import db_add_score, db_query_by_id, word_name_query_item
from config import *

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class ServerError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        error_dict = dict(self.payload or ())
        error_dict['result'] = self.message
        return error_dict


app = Flask(__name__)  # pylint: disable=invalid-name
app.debug = False

# 配置消息代理的路径
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# 要存储 Celery 任务的状态或运行结果时就必须要配置
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# app.config['CELERYD_MAX_TASKS_PER_CHILD'] = 20
# app.config['CELERY_IGNORE_RESULT'] = True
#
# #任务过期时间，单位为s，默认为一
# app.config['CELERY_TASK_RESULT_EXPIRE'] = 1000
# #backen缓存结果的数目，默认5000
# app.config['CELERY_MAX_CACHED_RESULT'] = 500


# 初始化Celery
celery_app = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# 将Flask中的配置直接传递给Celery
celery_app.conf.update(app.config)

# celery_app.conf.CELERYD_MAX_TASKS_PER_CHILD = 20
# celery_app.conf.CELERY_IGNORE_RESULT = True
#任务过期时间，单位为s，默认为一
celery_app.conf.CELERY_TASK_RESULT_EXPIRE = 1000
#backen缓存结果的数目，默认5000
celery_app.conf.CELERY_MAX_CACHED_RESULT = 200

control = Control(celery_app)

predictor = Predictor.by_name(PREDICTOR)()

@celery_app.task(bind=True)   #, ignore_result=True
def celery_predict(self, json_data):
    global predictor
    self.update_state(state='PROGRESS')
    prediction = predictor.predict_json(json_data)
    result = {}
    result['result'] = prediction
    return result

@celery_app.task(bind=True)   #, ignore_result=True
def celery_predict_single_char(self, json_data):
    self.update_state(state='PROGRESS')
    recog_predictor = Recognize_Predictor_batch()
    prediction = recog_predictor.predict_json(json_data)
    result = {}
    result['result'] = prediction
    return result


@app.errorhandler(ServerError)
def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/predict', methods=['POST'])
def predict() -> Response:  # pylint: disable=unused-variable
    """make a prediction using the specified model and return the results"""
    json_data = request.get_json()
    task = celery_predict.delay(json_data)
    print('task id: ', task.id)
    return jsonify({'state':task.state, 'location': url_for('taskstate', task_id=task.id)})

@app.route('/recognize', methods=['POST'])
def predict_single_char() -> Response:  # pylint: disable=unused-variable
    """make a prediction using the specified model and return the results"""
    json_data = request.get_json()
    task = celery_predict_single_char.delay(json_data)
    print('task id: ', task.id)
    return jsonify({'state':task.state, 'location': url_for('taskstate', task_id=task.id)})


@app.route('/status/<task_id>', methods=['GET'])
def taskstate(task_id):
    task = celery_predict.AsyncResult(task_id)

    if task.state == 'SUCCESS':
        response = {'state': task.state}
        if 'result' in task.info:
            response['result'] = task.info.get('result', [])
        else:
            response['result'] = []
    elif task.state == 'PENDING':  # 在等待
        response = {'state': task.state}
    else:
        response = {'state': task.state}
    return jsonify(response)


@app.route('/shareimg/<img_id>', methods=['GET'])
def sharedimg(img_id):
    base64_img = ""
    try:
        print('/shareimg/', img_id)
        data = db_query_by_id(img_id)
        img_path = data.get("IMG_PATH", '')
        lab_path = data.get("LAB_PATH", '')
        if len(data) == 0 or len(img_path) == 0:
            response = {'image': ''}
            return jsonify(response)
        if len(lab_path) != 0:
            boxes_list, text_list = load_boxes(lab_path)
            drawed_img = draw_bbox(img_path, np.array(boxes_list), text_list=text_list, font_size=20)
            base64_img = cvImg_to_base64(img_path, drawed_img)
        else:
            base64_img = cv2_to_base64(img_path)
    except:
        print('shareimg error.')
        base64_img = ""
    finally:
        response = {'image': base64_img}

    return jsonify(response)


#使正在等待的任务停止
@app.route('/revoke/<task_id>', methods=['GET'])
def taskrevoke(task_id):
    task = celery_predict.AsyncResult(task_id)
    control.revoke(task.id, terminate=True)
    print('task revoked: ', task.id)
    response = {'state': task.state}
    return jsonify(response)


#给识别结果打分
@app.route('/score', methods=['POST'])
def score():
    json_data = request.get_json()
    db_add_score(json_data)
    # {"imgid":23, "score":4.5}
    return jsonify({})

#文字释义
@app.route('/explainword/<word>', methods=['GET'])
def explainword(word):
    pass



if __name__ == "__main__":
    CORS(app)
    if len(SSL_KEY) != 0:
        http_server = WSGIServer(('0.0.0.0', PORT), app, keyfile=SSL_KEY, certfile=SSL_PEM)
    else:
        http_server = WSGIServer(('0.0.0.0', PORT), app)

    print(f"{PREDICTOR} is running on port. {PORT}")
    http_server.serve_forever()
