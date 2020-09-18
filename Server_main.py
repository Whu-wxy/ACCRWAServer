# coding=UTF-8<code>
from typing import List, Callable, TypeVar
import argparse
import json
import logging
import os
import sys

from flask import Flask, request, Response, jsonify
from flask_cors import CORS

from gevent import monkey
from gevent.pywsgi import WSGIServer

from predictor.Predictor import Predictor

from utils import check_for_gpu

#monkey.patch_all()   #设置多进程，但是模型应该一次只能处理一张图，

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
        error_dict['message'] = self.message
        return error_dict


def make_app(predictor: Predictor) -> Flask:

    # 服务命名为app
    app = Flask(__name__)  # pylint: disable=invalid-name

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    # @app.route('/')
    # def index() -> Response: # pylint: disable=unused-variable
    #     if static_dir is not None:
    #         return send_file(os.path.join(static_dir, 'index.html'))
    #     else:
    #         html = _html(title, field_names)
    #         return Response(response=html, status=200)

    @app.route('/predict', methods=['POST', 'OPTIONS'])
    def predict() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        #print("request:", request.data)
        #data = request.get_json()
        upload_file = request.files['file']

        prediction = predictor.predict_json(data=upload_file)

        return jsonify(prediction)

    return app


def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda)

    return Predictor.from_args(args.predictor)



def main(args):

    parser = argparse.ArgumentParser(description='Serve up a simple model')

    parser.add_argument('--cuda', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('--port', type=int, default=8000, help='port of the server')
    parser.add_argument('--predictor', type=str, required=True, help='name of predictor')


    args = parser.parse_args(args)

    predictor = _get_predictor(args)

    app = make_app(predictor=predictor)
    CORS(app)

    http_server = WSGIServer(('0.0.0.0', args.port), app)
    print(f"Serving demo on port {args.port}")
    http_server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
