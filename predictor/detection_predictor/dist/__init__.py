import subprocess
import os
import numpy as np
import cv2
import timeit


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile dis: {}'.format(BASE_DIR))

from .dist import dist_cpp


##
def dilate_alg(center, min_area=5, probs=None):
    center = np.array(center)
    label_num, label_img = cv2.connectedComponents(center.astype(np.uint8), connectivity=4)

    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label_img == label_idx) < min_area:
            label_img[label_img == label_idx] = 0
            continue

        score_i = np.mean(probs[label_img == label_idx])
        if score_i < 0.85:
            continue
        label_values.append(label_idx)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))  # 椭圆结构
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  # 椭圆结构
    for label_idx in range(1, label_num):
        label_i = np.where(label_img == label_idx, 255, 0)
        label_dilation = cv2.dilate(label_i.astype(np.uint8), kernel)
        bi_label_dilation = np.where(label_dilation == 255, 0, 1)
        label_dilation = np.where(label_dilation == 255, label_idx, 0)
        label_img = bi_label_dilation * label_img + label_dilation

    return np.array(label_img), label_values


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def decode(preds, scale, dmax=0.64, dmin=0.295, center_th=0.95, full_th=0.978):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :return: 最后的输出图和文本框
    """

    bi_region = preds[1, :, :]
    dist_map = preds[0, :, :]

    bi_region = sigmoid(bi_region)
    if len(bi_region.shape) == 3:
        bi_region = np.squeeze(bi_region)

    dist_map = sigmoid(dist_map)

    if len(dist_map.shape) == 3:
        dist_map = np.squeeze(dist_map)


    dist_map = dist_map + bi_region - 1
    region = np.where(dist_map >= dmin, 1, 0)
    center = np.where(dist_map >= dmax, 1, 0)

    area_threld = int(250*scale)
    pred = dist_cpp(center.astype(np.uint8), region.astype(np.uint8), bi_region, center_th, full_th, area_threld)

    bbox_list = []
    label_values = int(np.max(pred))
    for label_value in range(label_values+1):
        if label_value == 0:
            continue
        points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]

        rect = cv2.minAreaRect(points)
        # if rect[1][0] > rect[1][1]:
        #     if rect[1][1] <= 10*scale:
        #         continue
        # else:
        #     if rect[1][0] <= 10*scale:
        #         continue

        bbox = cv2.boxPoints(rect)

        bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])

    bbox_list = np.array(bbox_list).astype(int)
    if len(bbox_list):
        bbox_list = bbox_list / scale
    bbox_list = bbox_list.astype(int)
    bbox_list = bbox_list.tolist()
    return bbox_list

