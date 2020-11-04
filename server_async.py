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

from gevent.pywsgi import WSGIServer
from predictor.Predictor import Predictor
from utils import check_for_gpu, allowed_file
from celery import Celery
from celery.app.task import Task
from celery.app.control import Control
from Sqlite3.sqlite import db_add_score
from config import *

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


port = 8009

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


#使正在等待的任务停止
@app.route('/revoke/<task_id>', methods=['GET'])
def taskrevoke(task_id):
    task = celery_predict.AsyncResult(task_id)
    control.revoke(task.id, terminate=True)
    print('task revoked: ', task.id)
    response = {'state': task.state}
    return jsonify(response)


@app.route('/score', methods=['POST'])
def score():
    json_data = request.get_json()
    db_add_score(json_data)
    # {"imgid":23, "score":4.5}

    return jsonify({})


if __name__ == "__main__":
    CORS(app)
    if len(SSL_KEY) != 0:
        http_server = WSGIServer(('0.0.0.0', PORT), app, keyfile=SSL_KEY, certfile=SSL_PEM)
    else:
        http_server = WSGIServer(('0.0.0.0', PORT), app)

    print(f"{PREDICTOR} is running on port. {PORT}")
    http_server.serve_forever()
