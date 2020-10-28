PREDICTOR = 'ancient-chinese'    # detection  ancient-chinese  recognize recognize_batch  test

DETECTION_MODEL_PATH = '/home/beidou/PycharmProjects/ACCRWAServer/models/mobilenet.onnx'  #./models/mobilenet.onnx
LONG_SIZE = 2000    # None
IMG_SCALE = 1.5
POST_DB = True
UNCLIP_RATIO = 1.5
D_MAX = 0.6
DB_THRESH = 0.7

RECOGNITION_MODEL_PATH = '/home/beidou/PycharmProjects/ACCRWAServer/models/recognize'  #./models/

DB_PATH = './Sqlite3/database.db'

SAVE_ROOT_PATH = '../users_data'

FONT_PATH = '/home/beidou/PycharmProjects/ACCRWAServer/models/NotoSansCJK-Regular.ttc'