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
from predictor.detection_predictor.db_decode import DB_Decoder
from werkzeug.utils import secure_filename
from predictor.detection_predictor.detection_predictor import Detection_Predictor
from utils import *
import timeit
import copy

def demo():

	long_size = 1500
	img_path = './test.jpg'
	print(onnxruntime.get_device())


	detector = onnxruntime.InferenceSession("../models/dmnet.onnx")

	detector.get_modelmeta()
	first_input_name = detector.get_inputs()[0].name
	first_output_name = detector.get_outputs()[0].name

	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	h, w = img.shape[:2]

	scale = 1
	if long_size != None:
		scale = long_size / max(h, w)
		img2 = cv2.resize(img, None, fx=scale, fy=scale)

	# 将图片由(w,h)变为(1,img_channel,h,w)
	tensor = transforms.ToTensor()(img2)
	tensor = tensor.unsqueeze_(0)

	start = timeit.default_timer()
	result = detector.run([], {"input": tensor.cpu().numpy()})
	end = timeit.default_timer()
	print('model time: ', end - start)
	result = result[0]

	print(result.shape)

	start = timeit.default_timer()
	# boxes_list = dist_decode(result[0], scale, dmax=0.6, center_th=0.91, full_th=0.91)
	db = DB_Decoder(unclip_ratio=0.5)
	boxes_list = db.predict(result[0], scale, dmax=0.6, center_th=0.91)

	end = timeit.default_timer()
	print('decode time: ', end - start)

	# boxes_list = np.array(boxes_list)
	# final_img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
	# cv2.namedWindow("final_img", cv2.WINDOW_NORMAL)
	# cv2.imshow('final_img', final_img)
	# cv2.waitKey()

	#cv2.imwrite('./test_db_decode0.5.jpg', final_img)

	sorted_boxes_list = sorted_boxes(np.array(boxes_list))
	crop_imgs = []
	for index, box in enumerate(sorted_boxes_list):
		tmp_box = copy.deepcopy(box)
		partImg = get_rotate_crop_image(img, tmp_box.astype(np.float32))
		h, w = partImg.shape[:2]
		if min(h, w) < 7:
			continue
		crop_imgs.append(partImg)
		cv2.namedWindow("partImg", cv2.WINDOW_NORMAL)
		cv2.imshow('partImg', partImg)
		cv2.waitKey()




#@Predictor.register('ancient-chinese')
class AncientChinesePredictor(Predictor):

	def __init__(self):
		self.detector = Detection_Predictor()
		sefl.recognizer = None

		self._model_init()

	def _model_init(self):
		pass

	def _json_preprocessing(self, request):
		return self.detector._json_preprocessing(request)

	def predict(self, instance):
		try:
			boxes_list = self.detector.predict(instance)
			img = cv2.imread(instance[0])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			sorted_boxes_list = sorted_boxes(np.array(boxes_list))
			crop_imgs = []
			for index, box in enumerate(sorted_boxes_list):
				tmp_box = copy.deepcopy(box)
				partImg = get_rotate_crop_image(img, tmp_box.astype(np.float32))
				h, w = partImg.shape[:2]
				if min(h, w) < 10:
					continue
				crop_imgs.append(partImg)
		except:
			return []


	def _predict_instance(self, instance):
		return None



if __name__ == '__main__':
	demo()