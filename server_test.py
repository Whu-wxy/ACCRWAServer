import requests
import json

def cv2_to_base64(image):
	data = cv2.imencode('.jpg', image)[1]
	return base64.b64encode(data.tostring()).decode('utf8')

def post():
	URL = 'http://127.0.0.1:8009/predict'

	post_data = {
		"username": "12345",
		"lon": 130,
		"lat": 31,
		'imgname':'test.jpg'
	}

	img_path = 'test.jpg'
	# 发送HTTP请求
	post_data['image'] = {cv2_to_base64(cv2.imread(img_path))}
	headers = {"Content-type": "application/json"}

	# filename = "1.jpg"
	# files = {'file': (filename, open("../1.jpg", 'rb'), 'image/jpg')}
	#req = requests.post(URL, data=post_data, files=files)

	req = requests.post(url=URL, headers=headers, data=json.dumps(data))

	data = req.content.decode('utf-8')
	data = json.loads(data)
	print(data)


if __name__ == '__main__':
    post()

