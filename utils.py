from torch import cuda
from typing import Union, List, TypeVar, Type, Dict
from collections import defaultdict

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



import datetime
import os


# 检查文件扩展名
def allowed_file(filename):
    allowed_extension = ['jpg', 'png', 'JPG', 'bmp']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extension

def get_extention(image_name):
    return image_name.rsplit('.', 1)[1].lower()


def save_img(image_data, image_extention, root_save_path='../images/'):
    allowed_extension = ['jpg', 'png', 'JPG', 'bmp']
    if not image_extention in allowed_extension:
        image_extention = 'jpg'

    time_now = datetime.datetime.now()
    img_save_path = root_save_path + time_now.strftime("%Y-%m-%d")
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    image_name = str(len(os.listdir(img_save_path)) + 1) + '.' + image_extention
    cv2.imwrite(os.path.join(img_save_path, image_name), image_data)


def image_decode():
    img = base64.b64decode(str(request.form["file"]))
    image_data = np.fromstring(img, np.uint8)
    image_data = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image_data