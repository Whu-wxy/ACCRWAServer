PREDICTOR = 'ancient-chinese'    # detection  ancient-chinese  recognize recognize_batch  test
PORT = 8009

DETECTION_MODEL_PATH = '../data/mobilenet_our.onnx'  #./models/mobilenet.onnx
MAX_LONG_SIZE = 2500    # None
IMG_SCALE = 1.5
POST_DB = True
D_MAX = 0.56

#DB
UNCLIP_RATIO = 1.4
DB_THRESH = 0.7

#DIST
CENTER_TH = 0.8
FULL_TH = 0.8


RECOGNITION_MODEL_NAME = 'inception_v4'  # resnet_v2_50
RECOGNITION_MODEL_PATH = '../data/recognize/inception_cyy/'  #./models/
DICT_SIZE = 3756

RECOG_IMG_SHAPE = 235  #224


DB_PATH = './data/database.db'

SAVE_ROOT_PATH = '../users_data'

FONT_PATH = '../data/NotoSansCJK-Regular.ttc'

SSL_KEY =  '' # './data/SSL/4695946_www.72qier.icu.key'
SSL_PEM = '' # './data/SSL/4695946_www.72qier.icu.pem'