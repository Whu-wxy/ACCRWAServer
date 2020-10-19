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
import timeit
from utils import *

def demo():

	long_size = 2000
	img_path = './1.jpg'
	print(onnxruntime.get_device())


	session = onnxruntime.InferenceSession("../../models/dmnet.onnx")

	session.get_modelmeta()
	first_input_name = session.get_inputs()[0].name
	first_output_name = session.get_outputs()[0].name

	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	h, w = img.shape[:2]

	scale = 1
	if long_size != None:
		scale = long_size / max(h, w)
		img = cv2.resize(img, None, fx=scale, fy=scale)

	# 将图片由(w,h)变为(1,img_channel,h,w)
	tensor = transforms.ToTensor()(img)
	tensor = tensor.unsqueeze_(0)

	start = timeit.default_timer()
	result = session.run([], {"input": tensor.cpu().numpy()})
	end = timeit.default_timer()
	print('model time: ', end - start)
	result = result[0]

	print(result.shape)

	start = timeit.default_timer()
	#boxes_list = dist_decode(result[0], scale, dmax=0.6, center_th=0.91, full_th=0.91)

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



@Predictor.register('detection')
class Detection_Predictor(Predictor):

	def __init__(self):
		self.session = None
		self.long_size = 2000

		self._model_init()

	def _model_init(self):
		self.session = onnxruntime.InferenceSession("./models/dmnet.onnx")

	def _json_preprocessing(self, data):

		img_save_path, result_save_path = get_img_save_dir('../')

		file_path = ''
		label_path = ''
		try:
			file_name = secure_filename(data['imgname'])
			image = base64_to_cv2(data['image'])
			if allowed_file(file_name):
				file_path, new_file_name = save_img(image, file_name, img_save_path)
				label_path = os.path.join(result_save_path, new_file_name.split('.')[0] + '.txt')
				print('file saved to %s' % file_path)
		except:
			print('upload_file is empty!')
		finally:
			return [file_path, label_path]


	def predict(self, instance):
		try:
			if os.path.exists(instance[0]):
				img = cv2.imread(instance[0])
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

				h, w = img.shape[:2]

				scale = 1
				if self.long_size != None:
					scale = self.long_size / max(h, w)
					img = cv2.resize(img, None, fx=scale, fy=scale)

				# 将图片由(w,h)变为(1,img_channel,h,w)
				tensor = transforms.ToTensor()(img)
				tensor = tensor.unsqueeze_(0)

				try:
					start = timeit.default_timer()
					result = self.session.run([], {"input": tensor.cpu().numpy()})
					end = timeit.default_timer()
					print('model time: ', end-start)

					result = result[0]
					start = timeit.default_timer()
					boxes_list = dist_decode(result[0], scale)
					end = timeit.default_timer()
					print('decode time: ', end - start)

					return boxes_list  # {"result": [[...], [...], [...]] }
				except:
					return []
			else:
				print('image is not exist!')
				return []
		except:
			return []

	def get_draw_img(self, img_path, boxes_list):
		boxes_list = np.array(boxes_list)
		final_img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
		return cvImg_to_base64(img_path, final_img)

	def _predict_instance(self, instance):
		try:
			boxes_list = self.predict(instance)
			save_boxes(instance[1], boxes_list)
			return {"result": boxes_list}  # {"result": [[...], [...], [...]] }

		except:
			return {"result":[]}

#{"result":[{"position":[-1, -1, -1, -1], "text":"123"}]}

if __name__ == '__main__':
	demo()

	# img_save_path, result_save_path = get_img_save_dir('../../../')
	# print(img_save_path)
	# print(result_save_path)
