from torch import cuda
from typing import Union, List, TypeVar, Type, Dict
from collections import defaultdict
import time
import datetime
import os
import cv2
import base64
import numpy as np

def parse_cuda_device(cuda_device: Union[str, int, List[int]]) -> Union[int, List[int]]:
    """
    Disambiguates single GPU and multiple GPU settings for cuda_device param.
    """
    def from_list(strings):
        if len(strings) > 1:
            return [int(d) for d in strings]
        elif len(strings) == 1:
            return int(strings[0])
        else:
            return -1

    if isinstance(cuda_device, str):
        return from_list(re.split(r',\s*', cuda_device))
    elif isinstance(cuda_device, int):
        return cuda_device
    elif isinstance(cuda_device, list):
        return from_list(cuda_device)
    else:
        return int(cuda_device)  # type: ignore


def check_for_gpu(device_id: Union[int, list]):
    device_id = parse_cuda_device(device_id)
    if isinstance(device_id, list):
        for did in device_id:
            check_for_gpu(did)
    elif device_id is not None and device_id >= 0:
        num_devices_available = cuda.device_count()
        if num_devices_available == 0:
            raise ConfigurationError("Experiment specified a GPU but none is available;"
                                     " if you want to run on CPU use the override"
                                     " 'trainer.cuda_device=-1' in the json config file.")
        elif device_id >= num_devices_available:
            raise ConfigurationError(f"Experiment specified GPU device {device_id}"
                                     f" but there are only {num_devices_available} devices "
                                     f" available.")


def get_img_save_dir(root_path='../../../'):
    # 按日期创建文件夹存图片
    date_now = time.strftime("%Y%m%d", time.localtime())
    #root_path = os.path.join(root_path, 'users_data')
    save_path = os.path.join(root_path, date_now)
    img_save_path = os.path.join(save_path, 'imgs')
    result_save_path = os.path.join(save_path, 'results')
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    return img_save_path, result_save_path



T = TypeVar('T')

class Registrable():

    _registry: Dict[Type, Dict[str, Type]] = defaultdict(dict)
    default_implementation: str = None

    @classmethod
    def register(cls: Type[T], name: str):
        registry = Registrable._registry[cls]
        def add_subclass_to_registry(subclass: Type[T]):
            # Add to registry, raise an error if key has already been used.
            if name in registry:
                message = "Cannot register %s as %s; name already in use for %s" % (
                        name, cls.__name__, registry[name].__name__)
                raise ValueError(message)
            registry[name] = subclass
            return subclass
        return add_subclass_to_registry

    @classmethod
    def by_name(cls: Type[T], name: str) -> Type[T]:
        print(f"instantiating registered subclass {name} of {cls}")
        if name not in Registrable._registry[cls]:
            raise ValueError("%s is not a registered name for %s" % (name, cls.__name__))
        return Registrable._registry[cls].get(name)




ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'JPG'}


# 检查文件扩展名
def allowed_file(filename):
    return '.' in filename and get_extention(filename) in ALLOWED_EXTENSIONS

def get_extention(image_name):
    return image_name.rsplit('.', 1)[1].lower()



def save_img(image_data, image_name, img_save_path):
    time_now = datetime.datetime.now()
    if not os.path.exists(img_save_path):
        raise ValueError("img_save_path not exist!")

    file_names = os.listdir(img_save_path)
    max_val = 0
    if len(file_names) == 0:
        max_val = 0
    else:
        max_val = max([int(name.split('.')[0]) for name in file_names])
    image_name = str(max_val + 1) + '.' + get_extention(image_name)
    cv2.imwrite(os.path.join(img_save_path, image_name), image_data)

    return os.path.join(img_save_path, image_name), image_name


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def draw_bbox(img_path, result, color=(255, 0, 0),thickness=2, text_list = None):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        if len(point) == 4:
            cv2.line(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
            cv2.line(img_path, tuple(point[1]), tuple(point[2]), color, thickness)
            cv2.line(img_path, tuple(point[2]), tuple(point[3]), color, thickness)
            cv2.line(img_path, tuple(point[3]), tuple(point[0]), color, thickness)
        elif len(point) == 2:
            cv2.rectangle(img_path, tuple(point[0]), tuple(point[1]), color, thickness)
    return img_path

def save_boxes(save_path, boxes, text_list=None):
    lines = []
    for i, bbox in enumerate(boxes):
        line = ''
        for box in bbox:
            line += "%d, %d, " % (int(box[0]), int(box[1]))

        if text_list != None:
            line += text_list[i]
        else:
            line = line.rsplit(',', 1)[0]

        line += '\n'
        lines.append(line)

    with open(save_path, 'w') as f:
        for line in lines:
            f.write(line)


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i+1][0][1] - _boxes[i][0][1]) < 10 and \
            (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_rotate_crop_image(img, points):
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    img_crop_width = int(np.linalg.norm(points[0] - points[1]))
    img_crop_height = int(np.linalg.norm(points[0] - points[3]))
    if img_crop_height == 0 or img_crop_width == 0:
        return img_crop
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img_crop, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)

    return dst_img




def cv2_to_base64(img_path):
    extention = '.' + get_extention(img_path)
    data = cv2.imencode(extention, cv2.imread(img_path))[1]
    return base64.b64encode(data.tostring()).decode('utf-8')

def cvImg_to_base64(img_path, img):
    extention = '.' + get_extention(img_path)
    data = cv2.imencode(extention, img)[1]
    return base64.b64encode(data.tostring()).decode('utf-8')

def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf-8'))
    data = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image
