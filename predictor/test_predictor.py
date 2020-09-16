from predictor.Predictor import Predictor
import time

@Predictor.register('test')
class TestPredictor(Predictor):

	def __init__(self):
		self._model_init()

	def _model_init(self):
		pass

	def _json_preprocessing(self, data):
		print("receive data: ", data)
		time.sleep(3)
		return data

	def _predict_instance(self, instance):
		return {"result":[{"position":[0, 0, 10, 10], "text":"识别结果1"}]}

