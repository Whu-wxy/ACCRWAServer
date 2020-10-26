import  sys
sys.path.append('../../')
from predictor.Predictor import Predictor
import time
import os
from werkzeug.utils import secure_filename
import timeit
from utils import *
from config import *

import numpy as np
import random
import tensorflow as tf
import cv2
from predictor.recognizion_predictor.nets import nets_factory
import json
from tensorflow.python.training import saver as tf_saver

slim = tf.contrib.slim

def cv_imread(file_path):
	img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
	img = img.astype('uint8')
	cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return cv_img


def cv_preprocess_image(img, output_height, output_width):
	assert output_height == output_width
	img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	img[:, :, 0] = np.uint8((np.int32(img[:, :, 0]) + (180 + random.randrange(-9, 10))) % 180)
	img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
	rows, cols, ch = img.shape
	output_size = output_width

	def r():
		return (random.random() - 0.5) * 0.1 * output_size

	pts1 = np.float32([[0, 0], [cols, rows], [0, rows]])
	pts2 = np.float32([[r(), r()], [output_size + r(), output_size + r()], [r(), output_size + r()]])
	M = cv2.getAffineTransform(pts1, pts2)
	noize = np.random.normal(0, random.random() * (0.05 * 255), size=img.shape)
	img = np.array(img, dtype=np.float32) + noize
	img = cv2.warpAffine(img, M, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
	return img


def tf_preprocess_image(image, output_height, output_width):
	image = tf.expand_dims(image, 0)
	image = tf.image.resize_bilinear(image, [output_height, output_width], align_corners=False)
	image = tf.squeeze(image, axis=0)
	image = tf.to_float(image)
	return image


def get_tf_preprocess_image():
	def foo(image, output_height, output_width):
		image = tf_preprocess_image(
			image, output_height, output_width)
		return tf.image.per_image_standardization(image)

	return foo



#@Predictor.register('recognize')
class Recognize_Predictor(Predictor):

	def __init__(self):
		self.session = None
		self.ops = None
		self.cates = None

		self._model_init()

	def _model_init(self):
		self.session = tf.Session()

		image_preprocessing_fn = get_tf_preprocess_image()
		images_holder = tf.placeholder(tf.uint8, shape=(None, None, 3))
		network_fn = nets_factory.get_network_fn('resnet_v2_50', 3755, weight_decay=0.0, is_training=False)
		eval_image_size = 224
		images = [image_preprocessing_fn(images_holder, eval_image_size, eval_image_size)]
		eval_ops, _ = network_fn(images)
		self.ops = eval_ops
		variables_to_restore = slim.get_variables_to_restore()
		saver = tf_saver.Saver(variables_to_restore)
		saver.restore(self.session, os.path.join(RECOGNITION_MODEL_PATH, 'train_logs_resnet_v2_50', 'model.ckpt-100000'))

		with open(os.path.join(RECOGNITION_MODEL_PATH, 'cates.json')) as f:
			lines = f.read()
			self.cates = json.loads(lines.strip())

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
		# try:
			img = cv_preprocess_image(img, 224, 224)
			# try:
			images_holder = tf.placeholder(tf.uint8, shape=(None, None, 3))
			start = timeit.default_timer()
			logits = self.session.run(self.ops, feed_dict={images_holder: img})

			assert 3755 == logits.shape[1]
			logits = logits[:, :3755]
			explogits = np.exp(np.minimum(logits, 70))
			expsums = np.sum(explogits, axis=1)
			expsums.shape = (logits.shape[0], 1)
			expsums = np.repeat(expsums, 3755, axis=1)
			probs = explogits / expsums
			argsorts = np.argsort(-logits, axis=1)

			predictions = []
			probabilities = []
			pred = argsorts[0][:5]
			prob = probs[0][pred].tolist()
			pred = list(map(lambda i: self.cates[i]['text'], pred.tolist()))
			predictions.append(pred)
			probabilities.append(prob)
			for i in range(5):
				print('predictions', predictions[0][i], probabilities[0][i])

			end = timeit.default_timer()
			print('[recognize] model time: ', end-start)

			return ['1']
		# 	except:
		# 		return ['2']
		# except:
		# 	return ['3']

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
	RECOGNITION_MODEL_PATH = './products'

	sess = Recognize_Predictor()
	img = cv2.imread('./testimages/0.jpg')
	res = sess.predict(img)
	print(res)

	# img_save_path, result_save_path = get_img_save_dir('../../../')
	# print(img_save_path)
	# print(result_save_path)
