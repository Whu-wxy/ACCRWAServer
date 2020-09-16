from predictor.Predictor import Predictor

@Predictor.register('ancient-chinese')
class AncientChinesePredictor(Predictor):

	def __init__(self):
		self._model_init()

	def _model_init(self):
		pass

	def _json_preprocessing(self, data):
		pass

	def _predict_instance(self, instance):
		pass
