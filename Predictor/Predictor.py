import torch

class Predictor():

    def __init__(self) -> None:
        _model_init()

    def _model_init(self):
        raise NotImplementedError

    def _json_preprocessing(self, data):
        raise NotImplementedError

    def _predict_instance(self, instance):
        raise NotImplementedError

    def predict_json(self, data):
        instance = _json_preprocessing(data)
        return _predict_instance(instance)