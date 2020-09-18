import requests
import json


def post():
	URL = 'http://127.0.0.1:8009/predict'
	filename = "img.jpg"
	files = {'file': (filename, open("../img.jpg", 'rb'), 'image/jpg')}
	req = requests.post(URL, files=files)
	data = req.content.decode('utf-8')
	data = json.loads(data)
	print(data)


if __name__ == '__main__':
    post()

