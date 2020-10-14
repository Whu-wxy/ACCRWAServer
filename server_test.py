import requests
import json


def post():
	URL = 'http://127.0.0.1:8009/predict'

	post_data = {
		"username": "12345",
		"lon": 130,
		"lat": 31,

	}
	filename = "1.jpg"
	files = {'file': (filename, open("../1.jpg", 'rb'), 'image/jpg')}

	headers = {'Content-Type': 'multipart/form-data; boundary=deecd88db1c3b9d6bc1dcb3c36d19b25'}
	#headers = {'Content-Type': 'application/x-www-form-urlencoded'}

	req = requests.post(URL, data=post_data, files=files)
	data = req.content.decode('utf-8')
	data = json.loads(data)
	print(data)


if __name__ == '__main__':
    post()

