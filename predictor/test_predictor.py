from predictor.Predictor import Predictor
import time
import os
from werkzeug.utils import secure_filename
from utils import  allowed_file, save_img, base64_to_cv2, get_img_save_dir
from flask import request

@Predictor.register('test')
class TestPredictor(Predictor):

	def __init__(self):
		self._model_init()

	def _model_init(self):
		pass

	def _json_preprocessing(self, data):
		print("receive data: ", data)
		time.sleep(10)
		return [1, 1, 0, 0]

		# img_save_path, result_save_path = get_img_save_dir('../')
		#
		# try:
		# 	file_name = secure_filename(data['imgname'])
		# 	image = base64_to_cv2(data['image'])
		# 	if allowed_file(file_name):
		# 		file_name = "123_"+file_name
		# 		save_img(image, file_name, img_save_path)
		# 		print('file saved to %s' % img_save_path)
		# 	return [0, 0, 10, 0, 10, 10, 0, 10]
		# except:
		# 	return []


	def _predict_instance(self, instance):
		print('after sleep: ', instance)
		return {"result":[{"position":instance, "text":"识别结果1"}]}

