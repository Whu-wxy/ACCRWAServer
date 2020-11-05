import os
import numpy as np
import cv2
import timeit
import pyclipper
from shapely.geometry import Polygon

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


class DB_Decoder():
	def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5):
		self.min_size = 3
		self.thresh = thresh
		self.box_thresh = box_thresh
		self.max_candidates = max_candidates
		self.unclip_ratio = unclip_ratio

	def predict(self, preds, scale, dmax=0.64, out_polygon=False):
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
		center = np.where(dist_map >= dmax, 1, 0)

		if out_polygon:
			boxes, scores = self.polygons_from_bitmap(bi_region, center)
		else:
			boxes, scores = self.boxes_from_bitmap(bi_region, center)

		if len(boxes):
			boxes = boxes*1.0 / scale
		boxes = boxes.astype(int)
		boxes = boxes.tolist()
		return boxes

		# area_threld = int(250*scale)
		#
		# bbox_list = []
		# label_values = int(np.max(pred))
		# for label_value in range(label_values+1):
		# 	if label_value == 0:
		# 		continue
		# 	points = np.array(np.where(pred == label_value)).transpose((1, 0))[:, ::-1]
		#
		# 	rect = cv2.minAreaRect(points)
		# 	# if rect[1][0] > rect[1][1]:
		# 	#     if rect[1][1] <= 10*scale:
		# 	#         continue
		# 	# else:
		# 	#     if rect[1][0] <= 10*scale:
		# 	#         continue
		#
		# 	bbox = cv2.boxPoints(rect)
		#
		# 	bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
		#
		# bbox_list = np.array(bbox_list).astype(int)
		# if len(bbox_list):
		# 	bbox_list = bbox_list / scale
		# bbox_list = bbox_list.astype(int)
		# bbox_list = bbox_list.tolist()
		# return bbox_list




	def polygons_from_bitmap(self, pred, bitmap):
		'''
		_bitmap: single map with shape (H, W),
			whose values are binarized as {0, 1}
		'''

		assert len(bitmap.shape) == 2
		# bitmap = _bitmap.cpu().numpy()  # The first channel
		# pred = pred.cpu().detach().numpy()
		height, width = bitmap.shape
		boxes = []
		scores = []

		_, contours, _  = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		for contour in contours[:self.max_candidates]:
			epsilon = 0.005 * cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, epsilon, True)
			points = approx.reshape((-1, 2))
			if points.shape[0] < 4:
				continue

			score = self.box_score_fast(pred, contour.squeeze(1))
			if self.box_thresh > score:
				continue

			if points.shape[0] > 2:
				box = self.unclip(points, unclip_ratio=self.unclip_ratio)
				if len(box) > 1:
					continue
			else:
				continue
			box = box.reshape(-1, 2)
			_, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)), height, width)
			if sside < self.min_size + 2:
				continue

			boxes.append(box)
			scores.append(score)
		return boxes, scores

	def boxes_from_bitmap(self, pred, bitmap):
		'''
		_bitmap: single map with shape (H, W),
			whose values are binarized as {0, 1}
		'''

		height, width = bitmap.shape
		_, contours, _  = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		num_contours = min(len(contours), self.max_candidates)
		boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
		scores = np.zeros((num_contours,), dtype=np.float32)

		for index in range(num_contours):
			contour = contours[index].squeeze(1)
			points, sside = self.get_mini_boxes(contour)
			if sside < self.min_size:
				continue
			points = np.array(points)
			score = self.box_score_fast(pred, contour)
			if self.box_thresh > score:
				continue

			box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
			box, sside = self.get_mini_boxes(box, height, width)
			if sside < self.min_size + 2:
				continue

			box = np.array(box)

			boxes[index, :, :] = box.astype(np.int16)
			scores[index] = score
		return boxes, scores

	def unclip(self, box, unclip_ratio=1.5):
		poly = Polygon(box)
		distance = poly.area * unclip_ratio / poly.length
		offset = pyclipper.PyclipperOffset()
		offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
		expanded = np.array(offset.Execute(distance))
		return expanded

	def get_mini_boxes(self, contour, height=None, width=None):
		bounding_box = cv2.minAreaRect(contour)
		points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

		index_1, index_2, index_3, index_4 = 0, 1, 2, 3
		if points[1][1] > points[0][1]:
			index_1 = 0
			index_4 = 1
		else:
			index_1 = 1
			index_4 = 0
		if points[3][1] > points[2][1]:
			index_2 = 2
			index_3 = 3
		else:
			index_2 = 3
			index_3 = 2

		box = [points[index_1], points[index_2], points[index_3], points[index_4]]

		if width != None:
			for i, pt in enumerate(box):
				pt[0] = 0 if pt[0]<0 else pt[0]
				pt[1] = 0 if pt[1]<0 else pt[1]
				pt[0] = width-1 if pt[0]>=width else pt[0]
				pt[1] = height-1 if pt[1]>=height else pt[1]
				box[i] = pt

		return box, min(bounding_box[1])

	def box_score_fast(self, bitmap, _box):
		h, w = bitmap.shape[:2]
		box = _box.copy()
		xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
		xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
		ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
		ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

		mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
		box[:, 0] = box[:, 0] - xmin
		box[:, 1] = box[:, 1] - ymin
		cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
		return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
