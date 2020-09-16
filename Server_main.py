# coding=UTF-8<code>
from typing import List, Callable, TypeVar
import argparse
import json
import logging
import os
import sys

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from Predictor import Predictor
T = TypeVar('T', Predictor)

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


def make_app(predictor):

    # 服务命名为app
    app = Flask(__name__)  # pylint: disable=invalid-name

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error):  # pylint: disable=unused-variable
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
    def predict(predictor):  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()

        prediction = predictor.predict_json(data)

        log_blob = {"outputs": prediction}
        logger.info("prediction: %s", json.dumps(log_blob))

        return jsonify(prediction)

    return app



def main(args):

    parser = argparse.ArgumentParser(description='Serve up a simple model')

    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('--port', type=int, default=8000, help='port to serve the demo on')

    args = parser.parse_args(args)

    #predictor = _get_predictor(args)

    predictor = Predictor()
    app = make_app(predictor=predictor)
    CORS(app)

    http_server = WSGIServer(('0.0.0.0', args.port), app)
    print("Model loaded, serving demo on port "+args.port)
    http_server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
