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
from predictor.detection_predictor.db_decode import DB_Decoder
from werkzeug.utils import secure_filename
from predictor.detection_predictor.detection_predictor import Detection_Predictor
from Sqlite3.sqlite import db_add_item
from utils import *
import timeit
import copy
from config import *
from predictor.recognizion_predictor.recognize_predictor import Recognize_Predictor_batch, Recognize_Predictor

def demo():

	long_size = 1600
	img_path = './test3.jpg'
	print(onnxruntime.get_device())


	detector = onnxruntime.InferenceSession(DETECTION_MODEL_PATH)
	recognizor = Recognize_Predictor()

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
	if POST_DB:
		db = DB_Decoder(unclip_ratio=1.5)
		boxes_list = db.predict(result[0], scale, dmax=0.6, center_th=0.91)
	else:
		from predictor.detection_predictor.dist import decode as dist_decode
		boxes_list = dist_decode(result[0], scale)

	end = timeit.default_timer()
	print('decode time: ', end - start)

	final_img = draw_bbox(img_path, np.array(boxes_list), color=(0, 0, 255))
	cv2.namedWindow("final_img", cv2.WINDOW_NORMAL)
	cv2.imshow('final_img', final_img)
	cv2.waitKey()

	#cv2.imwrite('./test_db_decode0.5.jpg', final_img)

	result = {}
	result_list = []
	for box in boxes_list:
		result_dict = {}
		result_dict["position"] = box
		partImg = get_rotate_crop_image(img, np.array(box).astype(np.float32))
		h, w = partImg.shape[:2]
		print(partImg.shape)
		if min(h, w) < 10:
			continue

		texts = recognizor.predict(partImg)
		print(texts)
		cv2.namedWindow("partImg", cv2.WINDOW_NORMAL)
		cv2.imshow('partImg', partImg)
		cv2.waitKey()
		result_dict["text"] = "1"
		result_list.append(result_dict)
	result["result"] = result_list
	result["image"] = "asdfsdfsdwerwvdf"
	print(result)



@Predictor.register('ancient-chinese')
class AncientChinesePredictor(Predictor):
	def __init__(self):
		self.detector = Detection_Predictor()
		self.recognizor = Recognize_Predictor_batch()

		self._model_init()

	def _model_init(self):
		pass

	def _json_preprocessing(self, data):
		return self.detector._json_preprocessing(data)

	def predict(self, img, gt_path=None):
		try:
			boxes_list = []
			if gt_path == None:
				boxes_list = self.detector.predict(img)
			else:
				with open(gt_path, 'r') as f:
					for line in f.readlines():
						params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
						x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
						boxes_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

			text_list = []
			crop_img_list = []
			new_box_list = []
			for box in boxes_list:
				try:
					crop_img = get_rotate_crop_image(img, np.array(box).astype(np.float32))
				except:
					print('crop error')
					continue
				h, w = crop_img.shape[:2]
				if min(h, w) < 10:
					continue
				new_box_list.append(box)
				crop_img_list.append(crop_img)
				#是否需要把图像块存起来

			if len(crop_img_list) == 0:
				return [], []
			texts, prob_list = self.recognizor.predict(crop_img_list)
			for text in texts:
				if len(text) != 0:
					text = text[0]
				else:
					text = ""
				text_list.append(text)
			return new_box_list, text_list
		except:
			print('error in ancient_chinese 1.')
			return [], []


	def predict_origin(self, img):
		try:
			boxes_list = self.detector.predict(img)

			text_list = []
			for box in boxes_list:
				crop_img = get_rotate_crop_image(img, np.array(box).astype(np.float32))
				h, w = crop_img.shape[:2]
				if min(h, w) < 10:
					continue
				texts = self.recognizor.predict(crop_img)
				print(texts)
				if len(texts) != 0:
					text = texts[0][0]
				else:
					text = ""

				#是否需要把图像块存起来

				text_list.append(text)
			return boxes_list, text_list
		except:
			return [], []

	def get_draw_img(self, img_path, boxes_list, text_list=None, thickness=2):
		boxes_list = np.array(boxes_list)
		final_img = draw_bbox(img_path, boxes_list, color=(0, 0, 255), text_list=text_list, thickness=thickness)
		return cvImg_to_base64(img_path, final_img)


	def _predict_instance(self, instance):
		try:
			img = None
			if len(instance)!=0 and os.path.exists(instance[0]):
				img = cv2.imread(instance[0])
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			else:
				print('image is not exist!')
				return {"boxes": [], "image": "", "imgid": -1}
			boxes_list, text_list = self.predict(img)
			save_boxes(instance[1], boxes_list, text_list)

			# add to DB
			json_data = instance[2]
			json_data['img_path']=instance[0]
			json_data['lab_path'] = instance[1]
			id = db_add_item(json_data)
			#
			if len(boxes_list) == 0:
				return {"boxes": [], "image": "", "imgid": -1}

			base64_img = self.get_draw_img(instance[0], boxes_list, text_list)
			result = {}
			result_list = []
			for box, text in zip(boxes_list, text_list):
				result_dict = {}
				result_dict["boxes"] = box
				result_dict["text"] = text
				result_list.append(result_dict)
			result["result"] = result_list
			result["image"] = base64_img
			result["imgid"] = id
			return result
		except:
			print('error in ancient_chinese 2.')
			return {"boxes": [], "image":"", "imgid": -1}


#{'result': [{'position': [[290, 239], [360, 249], [356, 270], [287, 259]], 'text': '1'},
#            {'position': [[358, 250], [423, 164], [451, 186], [386, 271]], 'text': '1'}],
#            'image': 'asdfsdfsdwerwvdf', 'imgid': 32}



if __name__ == '__main__':
	# demo()

	sess = AncientChinesePredictor()

	imgs = os.listdir('./img')
	save_dir = './save'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	for im in imgs:
		path = os.path.join('./img', im)
		gt_path = os.path.join('./gt', im.split('.')[0]+'.txt')
		img = cv2.imread(path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		boxes_list, text_list = sess.predict(img, gt_path)
		base64_img = sess.get_draw_img(path, boxes_list, text_list)
		img2 = base64_to_cv2(base64_img)
		# cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
		# cv2.imshow('img2', img2)
		# cv2.waitKey()
		cv2.imwrite(os.path.join(save_dir, im), img2)


	# result = {}
	# result_list = []
	# for box, text in zip(boxes_list, text_list):
	# 	result_dict = {}
	# 	result_dict["boxes"] = box
	# 	result_dict["text"] = text
	# 	result_list.append(result_dict)
	# result["result"] = result_list
	# result["image"] = "sdfsdfsd"
	# result["imgid"] = -1
	# print(result)

	print('finished')