from predictor.Predictor import Predictor
import time
import os
from werkzeug.utils import secure_filename
from flask import request

@Predictor.register('test')
class TestPredictor(Predictor):

	def __init__(self):
		self._model_init()

	def _model_init(self):
		pass

	def _json_preprocessing(self, request):
		# print("receive data: ", data)
		# time.sleep(3)

		upload_file = request.files['file']

		old_file_name = secure_filename(upload_file.filename)
		print(old_file_name)
		if upload_file:
			file_path = os.path.join('./', "123_"+old_file_name)
			upload_file.save(file_path)
			print('file saved to %s' % file_path)
			print('success!')
			return 'success'
		else:
			return 'failed'

		return data

	def _predict_instance(self, instance):
		return {"result":[{"position":[0, 0, 10, 0, 10, 10, 0, 10], "text":"识别结果1"}]}

