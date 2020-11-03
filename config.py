PREDICTOR = 'ancient-chinese'    # detection  ancient-chinese  recognize recognize_batch  test

DETECTION_MODEL_PATH = '/home/beidou/PycharmProjects/ACCRWAServer/models/mobilenet.onnx'  #./models/mobilenet.onnx
MAX_LONG_SIZE = 2500    # None
IMG_SCALE = 1.5
POST_DB = False
D_MAX = 0.56

#DB
UNCLIP_RATIO = 1.5
DB_THRESH = 0.7

#DIST
CENTER_TH = 0.8
FULL_TH = 0.8


RECOGNITION_MODEL_PATH = '/home/beidou/PycharmProjects/ACCRWAServer/models/recognize'  #./models/
RECOG_IMG_SHAPE = 224  #224


DB_PATH = './Sqlite3/database.db'

SAVE_ROOT_PATH = '../users_data'

FONT_PATH = '/home/beidou/PycharmProjects/ACCRWAServer/models/NotoSansCJK-Regular.ttc'