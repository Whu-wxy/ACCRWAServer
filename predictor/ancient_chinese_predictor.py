import  sys
sys.path.append('../../')
from predictor.Predictor import Predictor
import time
from flask import request
import os
import onnxruntime
import cv2
from torchvision import transforms
import numpy as np
from predictor.detection_predictor.dist import decode as dist_decode
from utils import get_img_save_dir
from werkzeug.utils import secure_filename
from predictor.detection_predictor.detection_predictor import Detection_Predictor

@Predictor.register('ancient-chinese')
class AncientChinesePredictor(Predictor):

	def __init__(self):
		self.detector = Detection_Predictor()

		self._model_init()

	def _model_init(self):
		pass

	def _json_preprocessing(self, request):
		return self.detector._json_preprocessing(request)


	def _predict_instance(self, instance):
		positions = self.detector.predict(instance)
		print(positions)
