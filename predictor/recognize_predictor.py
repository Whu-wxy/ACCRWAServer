import  sys
sys.path.append('../../')
from predictor.Predictor import Predictor
import time
import os
import onnxruntime
import cv2
from torchvision import transforms
import numpy as np

from werkzeug.utils import secure_filename
import timeit
from utils import *
from config import *

def demo():
	pass



@Predictor.register('recognize')
class Recognize_Predictor(Predictor):

	def __init__(self):
		self.session = None
		self.long_size = 2000

		self._model_init()

	def _model_init(self):
		self.session = onnxruntime.InferenceSession(RECOGNITION_MODEL_PATH)

	def _json_preprocessing(self, data):

		img_save_path, result_save_path = get_img_save_dir(os.path.join(SAVE_ROOT_PATH, 'recognition'))

		file_path = ''
		label_path = ''
		try:
			pass
		# file_name = secure_filename(data['imgname'])
		# image = base64_to_cv2(data['image'])
		# if allowed_file(file_name):
		# 	file_path, new_file_name = save_img(image, file_name, img_save_path)
		# 	label_path = os.path.join(result_save_path, new_file_name.split('.')[0] + '.txt')
		# 	print('file saved to %s' % file_path)
		#
		# 	#data['img_path'] = file_path
		# else:
		# 	return []
		except:
			print('upload_file is empty!')
		finally:
			return [file_path, label_path, data]


	def predict(self, img):
		#这里改成直接传图片比较好，方便在ancient_chine里调用
		try:
			h, w = img.shape[:2]

			scale = 1
			if self.long_size != None:
				scale = self.long_size / max(h, w)
				img = cv2.resize(img, None, fx=scale, fy=scale)

			# 将图片由(w,h)变为(1,img_channel,h,w)
			tensor = transforms.ToTensor()(img)
			tensor = tensor.unsqueeze_(0)

			try:
				text_list = []
				start = timeit.default_timer()
				result = self.session.run([], {"input": tensor.cpu().numpy()})
				end = timeit.default_timer()
				print('[recognize] model time: ', end-start)

				return text_list
			except:
				return []
		except:
			return []

	def _predict_instance(self, instance):
		#在这里得到结果之后，对图片进行重命名，为空的字符串则不改名字

		try:
			img = None
			if os.path.exists(instance[0]):
				img = cv2.imread(instance[0])
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			else:
				print('image is not exist!')
				return {"boxes": []}
			text_list = self.predict(img)
			os.rename(instance[0], "XXXX")

			# add to DB
			json_data = instance[2]
			#

			if len(text_list) == 0:
				return {"boxes": []}

			return {"boxes": text_list}  # {"result": [[...], [...], [...]] }

		except:
			return {"boxes":text_list}

#{"result":[ ["A", 0.8], ["B", 0.2] ]}

if __name__ == '__main__':
	demo()

	# img_save_path, result_save_path = get_img_save_dir('../../../')
	# print(img_save_path)
	# print(result_save_path)