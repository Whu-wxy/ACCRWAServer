PREDICTOR = 'ancient-chinese'    # detection  ancient-chinese  recognize recognize_batch  test
PORT = 8009    # 端口

### 检测模型参数
DETECTION_MODEL_PATH = '../data/mobilenet_best_best.onnx'  #./models/mobilenet.onnx
MAX_LONG_SIZE = 2500    # 如果长边放大IMG_SCALE倍后超过这个数，则长边最多能放大到MAX_LONG_SIZE
IMG_SCALE = 1.5
POST_DB = True # 后处理用DB的（速度更快）还是用距离图方法的后处理
D_MAX = 0.5    #0.56-1.4   用于调整中心线宽度，和UNCLIP_RATIO相搭配

#DB
UNCLIP_RATIO = 1
DB_THRESH = 0.95

#DIST   置信度阈值-中心线和完整区域，0-1，越低召回率越高，准确率越低
CENTER_TH = 0.8
FULL_TH = 0.8
###


### 识别模型参数
RECOGNITION_MODEL_NAME = 'inception_v4'  # resnet_v2_50
RECOGNITION_MODEL_PATH = '../data/recognize/inception_cyy/'  #./models/
DICT_SIZE = 3756

RECOG_IMG_SHAPE = 235  #224
RECOG_GROUP_NUM = 100
###

### 数据库
DB_PATH = './data/database.db'
WORD_DB_PATH = './data/word.db'
### 用户图片保存路径
SAVE_ROOT_PATH = '../users_data'
### 用于在图片上绘制简体汉字结果的字体文件
FONT_PATH = './data/NotoSansCJK-Regular.ttc'
### 用于发送给用户各种字体的图片

WORD_IMG_PATH = './data/font_examples'

### SSL证书
SSL_KEY =  '' # './data/SSL/whudcil.com.cn_key.key'
SSL_PEM = '' # './data/SSL/whudcil.com.cn_chain.crt'
###

import os
if not os.path.exists(SAVE_ROOT_PATH):
    os.makedirs(SAVE_ROOT_PATH)
