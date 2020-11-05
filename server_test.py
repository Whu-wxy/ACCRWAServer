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
	URL = 'http://121.37.141.237:8009/predict'


	post_data = {
		"username": "12345667890",
		"lon": 131,
		"lat": 32,
		'imgname':'test4.jpg',
		# 'img_path':'../../sdfs',
		# 'label_path':'../../sdfsdf'
	}

	img_path = './test4.jpg'

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


def status(url):
	URL = 'http://127.0.0.1:8009' +url
	# URL = 'http://119.3.124.157:8009' +url
	URL = 'https://www.72qier.icu:8009' +url
	URL = 'http://121.37.141.237:8009' +url


	req = requests.get(url=URL)

	data = req.content.decode('utf-8')
	data = json.loads(data)
	print(data)

	return data


def long_test():
	for i in range(7200):
		time.sleep(2)
		data = post()


if __name__ == '__main__':
	# long_test()

	# for i in range(10):
	# 	data = post()
	data1 = post()
	# data2 = post()
	# data3 = post()
	# data4 = post()
	#
	for i in range(30):
		time.sleep(2)
		res = status(data1['location'])
		if res['state'] == 'SUCCESS':
			break

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


