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


# image_preprocessing_fn = get_tf_preprocess_image()
# image_holder = tf.placeholder(tf.uint8, shape=(None, None, 3))
# network_fn = nets_factory.get_network_fn('resnet_v2_50', 3755, weight_decay=0.0, is_training=False)
# images = [image_preprocessing_fn(image_holder, 224, 224)]
# eval_ops, _ = network_fn(images)
# variables_to_restore = slim.get_variables_to_restore()
#
# session = tf.Session()
#
# saver = tf_saver.Saver(variables_to_restore)
# saver.restore(session, os.path.join(RECOGNITION_MODEL_PATH, 'train_logs_resnet_v2_50', 'model.ckpt-100000'))


@Predictor.register('recognize')
class Recognize_Predictor(Predictor):

	def __init__(self):
		self.session = None
		self.ops = None
		self.cates = None
		self.image_holder = None
		self.graph = tf.Graph()

		self._model_init()

	def _model_init(self):
		with self.graph.as_default():
			self.session = tf.Session()

			image_preprocessing_fn = get_tf_preprocess_image()
			self.image_holder = tf.placeholder(tf.uint8, shape=(None, None, 3))
			network_fn = nets_factory.get_network_fn('resnet_v2_50', 3755, weight_decay=0.0, is_training=False)
			images = [image_preprocessing_fn(self.image_holder, 224, 224)]
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
			file_name = secure_filename(data['imgname'])
			image = base64_to_cv2(data['image'])
			if allowed_file(file_name):
				file_path, new_file_name = save_img(image, file_name, img_save_path)
				label_path = os.path.join(result_save_path, new_file_name.split('.')[0] + '.txt')
				print('file saved to %s' % file_path)

				#data['img_path'] = file_path
			else:
				return []
		except:
			print('upload_file is empty!')
		finally:
			return [file_path, label_path, data]


	def predict(self, img):
		#这里改成直接传图片比较好，方便在ancient_chine里调用
		try:
			img = cv_preprocess_image(img, 224, 224)
			# try:
			start = timeit.default_timer()

			# with tf.Session() as session:
			# 	variables_to_restore = slim.get_variables_to_restore()
			# 	saver = tf_saver.Saver(variables_to_restore)
			# 	saver.restore(session, os.path.join(RECOGNITION_MODEL_PATH, 'train_logs_resnet_v2_50', 'model.ckpt-100000'))
			# 	logits = session.run(self.ops, feed_dict={self.image_holder: img})

			# global session, eval_ops, image_holder
			# logits = session.run(eval_ops, feed_dict={image_holder: img})


			logits = self.session.run(self.ops, feed_dict={self.image_holder: img})

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

			result = []
			for i in range(5):
				result.append([str(predictions[0][i]), float(probabilities[0][i])])
				# print('predictions', predictions[0][i], probabilities[0][i])

			end = timeit.default_timer()
			print('[recognize] model time: ', end-start)

			return result
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
			if len(text_list) != 0:
				os.rename(instance[0], text_list[0][0])

			# add to DB
			json_data = instance[2]
			#

			if len(text_list) == 0:
				return {"text": []}

			return {"text": text_list}

		except:
			return {"text": text_list}




@Predictor.register('recognize_batch')
class Recognize_Predictor_batch(Predictor):

	def __init__(self):
		self.cates = None

		self._model_init()

	def _model_init(self):
		with open(os.path.join(RECOGNITION_MODEL_PATH, 'cates.json')) as f:
			lines = f.read()
			self.cates = json.loads(lines.strip())

	def _json_preprocessing(self, data):

		img_save_path, result_save_path = get_img_save_dir(os.path.join(SAVE_ROOT_PATH, 'recognition'))

		file_path = ''
		label_path = ''
		try:
			file_name = secure_filename(data['imgname'])
			image = base64_to_cv2(data['image'])
			if allowed_file(file_name):
				file_path, new_file_name = save_img(image, file_name, img_save_path)
				label_path = os.path.join(result_save_path, new_file_name.split('.')[0] + '.txt')
				print('file saved to %s' % file_path)

				#data['img_path'] = file_path
			else:
				return []
		except:
			print('upload_file is empty!')
		finally:
			return [file_path, label_path, data]


	def predict(self, img_list):
		#这里改成直接传图片比较好，方便在ancient_chine里调用
		# try:
			n = len(img_list)
			# for im in img_list:
			# 	cv2.namedWindow("img2", cv2.WINDOW_NORMAL)
			# 	cv2.imshow('img2', im)
			# 	cv2.waitKey()

			tf.reset_default_graph()

			image_preprocessing_fn = get_tf_preprocess_image()
			images_holder = [tf.placeholder(tf.uint8, shape=(None, None, 3)) for i in range(n)]
			network_fn = nets_factory.get_network_fn('resnet_v2_50', 3755, weight_decay=0.0, is_training=False)
		#inception_v4 resnet_v2_50

			images = [image_preprocessing_fn(images_holder[i], RECOG_IMG_SHAPE, RECOG_IMG_SHAPE) for i in range(n)]
			eval_ops, _ = network_fn(images)

			variables_to_restore = slim.get_variables_to_restore()

			img_list_temp = []
			for img in img_list:
				img = cv_preprocess_image(img, RECOG_IMG_SHAPE, RECOG_IMG_SHAPE)
				img = np.expand_dims(img, 0)
				img_list_temp.append(img)
			img_list_temp = np.concatenate(img_list_temp, axis=0)

			start = timeit.default_timer()
			with tf.Session() as session:
				saver = tf_saver.Saver(variables_to_restore)
				saver.restore(session, os.path.join(RECOGNITION_MODEL_PATH, 'train_logs_resnet_v2_50', 'model.ckpt-100000'))
				results = []
				lo = 0
				while lo != len(img_list_temp):
					hi = min(len(img_list_temp), lo + n)
					feed_data = img_list_temp[lo:hi]
					logits = session.run(eval_ops, feed_dict={images_holder[i]: feed_data[i] for i in range(n)})
					results.append(logits[:hi - lo])
					lo = hi
				logits = np.concatenate(results, axis=0)

			assert 3755 == logits.shape[1]
			logits = logits[:, :3755]
			explogits = np.exp(np.minimum(logits, 70))
			expsums = np.sum(explogits, axis=1)
			expsums.shape = (logits.shape[0], 1)
			expsums = np.repeat(expsums, 3755, axis=1)
			probs = explogits / expsums
			argsorts = np.argsort(-logits, axis=1)

			lo = 0
			predictions = []
			probabilities = []
			for i in range(n):
				pred = argsorts[lo][:5]
				prob = probs[lo][pred].tolist()
				pred = list(map(lambda i: self.cates[i]['text'], pred.tolist()))
				predictions.append(pred)
				probabilities.append(prob)
				lo += 1

			# for i in range(5):
			# 	print('predictions', predictions[0][i], probabilities[0][i])

			end = timeit.default_timer()
			print('[recognize] model time: ', end-start)

			return predictions, probabilities
		# except:
		# 	return [], []

	def _predict_instance(self, instance):
		#在这里得到结果之后，对图片进行重命名，为空的字符串则不改名字
		try:
			img = None
			if os.path.exists(instance[0]):
				img = cv2.imread(instance[0])
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			else:
				print('image is not exist!')
				return {"text": [], 'probs': []}
			text_list, prob_list = self.predict([img])
			if len(text_list) != 0:
				os.rename(instance[0], text_list[0][0])

			# add to DB
			json_data = instance[2]
			#

			if len(text_list) == 0:
				return {"text": [], 'probs': []}

			return {"text": text_list, 'probs': prob_list}

		except:
			return {"text": [], 'probs': []}


#{"text":[ ["A", 0.8], ["B", 0.2] ]}

if __name__ == '__main__':
	sess = Recognize_Predictor()
	img = cv2.imread('../../0.jpg')
	res = sess.predict(img)
	print(res)
	res = sess.predict(img)
	print(res)
	res = sess.predict(img)
	print(res)

	# img_save_path, result_save_path = get_img_save_dir('../../../')
	# print(img_save_path)
	# print(result_save_path)
