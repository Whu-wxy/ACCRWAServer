import requests
import json
import cv2
import base64
import numpy as np
import time

def cv2_to_base64(image):
	data = cv2.imencode('.jpg', image)[1].tostring()
	return base64.b64encode(data).decode('utf-8')

def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf-8'))
    data = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image



def post():
	URL = 'http://127.0.0.1:8009/predict'
	# URL = 'http://119.3.124.157:8009/predict'
	URL = 'https://www.72qier.icu:8009/predict'
	# URL = 'http://121.37.141.237:8009/predict'


	post_data = {
		"username": "12345667890",
		"lon": 131,
		"lat": 32,
		'imgname':'test3.jpg',
		# 'img_path':'../../sdfs',
		# 'label_path':'../../sdfsdf'
	}

	img_path = './test3.jpg'

	# 发送HTTP请求
	post_data['image'] = cv2_to_base64(cv2.imread(img_path))
	#img = base64_to_cv2(post_data['image'])
	# headers = {"Content-type": "application/json"}
	# req = requests.post(url=URL, headers=headers, data=json.dumps(post_data))

	headers = {"Content-type": "application/json"}
	req = requests.post(url=URL, headers=headers, data=json.dumps(post_data))


	# filename = "1.jpg"
	# files = {'file': (filename, open("../1.jpg", 'rb'), 'image/jpg')}
	#req = requests.post(URL, data=post_data, files=files)


	data = req.content.decode('utf-8')
	data = json.loads(data)
	print(data)

	return data

def post_recog():
	URL = 'http://127.0.0.1:8009/recognize'
	# URL = 'http://119.3.124.157:8009/recognize'
	# URL = 'https://www.72qier.icu:8009/recognize'
	# URL = 'http://121.37.141.237:8009/recognize'

	post_data = {
		"username": "12345667890",
		"lon": 131,
		"lat": 32,
		'imgname':'0.jpg',
		# 'img_path':'../../sdfs',
		# 'label_path':'../../sdfsdf'
	}

	img_path = './0.jpg'

	# 发送HTTP请求
	post_data['image'] = cv2_to_base64(cv2.imread(img_path))

	headers = {"Content-type": "application/json"}
	req = requests.post(url=URL, headers=headers, data=json.dumps(post_data))

	data = req.content.decode('utf-8')
	data = json.loads(data)
	print(data)

	return data


def status(url):
	URL = 'http://127.0.0.1:8009' +url
	# URL = 'http://119.3.124.157:8009' +url
	# URL = 'https://www.72qier.icu:8009' +url
	# URL = 'http://121.37.141.237:8009' +url


	req = requests.get(url=URL)

	data = req.content.decode('utf-8')
	data = json.loads(data)
	print(data)

	return data


def long_test():
	for i in range(7200):
		time.sleep(2)
		data = post()

def get_shareimg():
	# URL = 'http://127.0.0.1:8009/shareimg/113'

	URL = 'https://www.72qier.icu:8009/shareimg/639'

	req = requests.get(url=URL)

	data = req.content.decode('utf-8')
	print(data)
	data = json.loads(data)

	img = base64_to_cv2(data['image'])
	cv2.namedWindow("final_img", cv2.WINDOW_NORMAL)
	cv2.imshow('final_img', img)
	cv2.waitKey()


def get_explain():
	word = '中'
	# word = word.encode('utf-8')
	# print(str(word))
	URL = 'http://127.0.0.1:8009/explainword/' + word

	URL = 'https://www.72qier.icu:8009/explainword/' + word

	req = requests.get(url=URL)

	data = req.content.decode('utf-8')

	data = json.loads(data)
	print(data)


def get_word_imgs():
	word = '正'
	# word = word.encode('utf-8')
	# print(str(word))
	URL = 'http://127.0.0.1:8009/wordimgs/' + word + '/16' + '/478c9d2e0dd84280a2dd9586a4dad4c3'

	# URL = 'https://www.72qier.icu:8009/explainword/' + word

	req = requests.get(url=URL)

	data = req.content.decode('utf-8')

	data = json.loads(data)
	print(data['result'].keys())
	print(data)


if __name__ == '__main__':

	# long_test()

	# get_explain()

	# for i in range(10):
	# 	data = post()

	get_word_imgs()

	# data2 = post_recog()

	# data1 = post()
	# data2 = post()
	# data3 = post()
	# data4 = post()
	# #
	# get_shareimg()

	for i in range(30):
		time.sleep(2)
	# 	res = status(data1['location'])
	# 	res3 = status(data3['location'])
	# 	res2 = status(data2['location'])
	# 	res4 = status(data4['location'])

		# if res['state'] == 'SUCCESS':
		# 	break

	img = base64_to_cv2(res['result']['image'])
	cv2.namedWindow("final_img", cv2.WINDOW_NORMAL)
	cv2.imshow('final_img', img)
	cv2.waitKey()


		# status(data2['location'])
		# status(data3['location'])
		# status(data4['location'])

	# 	status(data4['location'].replace('status', 'revoke'))
	#
	# status(data1['location'].replace('status', 'revoke'))
	# status(data2['location'].replace('status', 'revoke'))
	# status(data3['location'].replace('status', 'revoke'))


