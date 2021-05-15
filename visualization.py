import cv2
import os
from utils import draw_bbox
import numpy as np

def visualization(base_dir, save_dir):
    date_dir = os.listdir(base_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for date in date_dir:
        if not os.path.exists(os.path.join(save_dir, date)):
            os.makedirs(os.path.join(save_dir, date))
        imgs = os.listdir(os.path.join(base_dir, date, 'imgs'))
        for im in imgs:
            path = os.path.join(base_dir, date, 'imgs', im)
            gt_path = os.path.join(base_dir, date, 'results', im.split('.')[0]+'.txt')
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes_list = []
            text_list = []
            with open(gt_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                    x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                    boxes_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    text_list.append(params[-1])
            img = draw_bbox(path, np.array(boxes_list, 'int32'), color=(0, 0, 255), text_list=text_list)
            cv2.imwrite(os.path.join(save_dir, date, im), img)


if __name__ == '__main__':
    visualization('F:\zzzzzz\detection', 'F:\zzzzzz\save')
    print('finish.')