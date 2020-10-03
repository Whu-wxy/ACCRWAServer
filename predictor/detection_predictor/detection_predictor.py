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

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def draw_bbox(img_path, result, color=(255, 0, 0),thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        if len(point) == 4:
            cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
            cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
            cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
            cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
        elif len(point) == 2:
            cv2.rectangle(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
    return img_path


def demo():

	long_size = 1300
	img_path = './test.jpg'
	print(onnxruntime.get_device())


	session = onnxruntime.InferenceSession("./dmnet.onnx")

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

	result = session.run([], {"input": tensor.cpu().numpy()})
	result = result[0]

	print(result.shape)

	# result = np.squeeze(result)
	# distance_map = result[0]
	# distance_map = sigmoid(distance_map)
	# cv2.namedWindow("dist_map", cv2.WINDOW_NORMAL)
	# cv2.imshow('dist_map', distance_map)
	# cv2.waitKey()

	preds_temp, boxes_list = dist_decode(result[0], scale)

	final_img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
	cv2.namedWindow("final_img", cv2.WINDOW_NORMAL)
	cv2.imshow('final_img', final_img)
	cv2.waitKey()



@Predictor.register('detection')
class Detection_Predictor(Predictor):

	def __init__(self):
		self.session = None

		self._model_init()

	def _model_init(self):
		self.session = onnxruntime.InferenceSession("./dmnet.onnx")

	def _json_preprocessing(self, upload_file):

		file_name = upload_file.filename

		img_save_path, result_save_path = get_img_save_dir('../../../')

		file_path = ''
		try:
			file_path = os.path.join(img_save_path, file_name)
			upload_file.save(file_path)
			print('file saved to %s' % file_path)
		except:
			print('upload_file is empty!')
		finally:
			return [file_path, result_save_path]


	def predict(self, instance):
		try:
			if os.path.exists(instance[0]):
				img = cv2.imread(instance[0])
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				h, w = img.shape[:2]

				scale = 1
				if long_size != None:
					scale = long_size / max(h, w)
					img = cv2.resize(img, None, fx=scale, fy=scale)

				# 将图片由(w,h)变为(1,img_channel,h,w)
				tensor = transforms.ToTensor()(img)
				tensor = tensor.unsqueeze_(0)

				result = session.run([], {"input": tensor.cpu().numpy()})
				result = result[0]
				preds_temp, boxes_list = dist_decode(result[0], scale)

				position = []
				for box in boxes_list:
					box = box.astype(int)
					if len(box) == 4:
						position.append([box[0], box[1], box[2], box[3]])

				return position  # {"result": [[...], [...], [...]] }
			else:
				print('image is not exist!')
		except:
			return []

	def _predict_instance(self, instance):
		try:
			position = self.predict(instance)
			return {"result": position}  # {"result": [[...], [...], [...]] }

		except:
			return {"result":[]}

#{"result":[{"position":[-1, -1, -1, -1], "text":"123"}]}

if __name__ == '__main__':
	#demo()

	img_save_path, result_save_path = get_img_save_dir('../../../')
	print(img_save_path)
	print(result_save_path)
