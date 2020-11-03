import torch
from utils import Registrable

class Predictor(Registrable):

    def __init__(self):
        self._model_init()

    def _model_init(self):
        pass

    def _json_preprocessing(self, data):
        raise NotImplementedError

    def _predict_instance(self, instance):
        raise NotImplementedError

    def predict_json(self, data):
        instance = self._json_preprocessing(data)
        return self._predict_instance(instance)

    @classmethod
    def from_args(cls, predictor_name: str = None):
        return Predictor.by_name(predictor_name)()
