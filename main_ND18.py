import cv2
import numpy as np

classes = "yolo/coco.names"
config = "yolo/yolov3.cfg"
weights = "yolo/yolov3.weights"

image = cv2.imread("res/plane.jpeg")

width = image.shape[1]
height = image.shape[0]
scale = 0.00392

names = open(classes).read().splitlines()

net = cv2.dnn.readNet(weights, config)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), True, crop=False)
get_blob = blob.reshape(blob.shape[2], blob.shape[3], blob.shape[1])


cv2.imshow("Blob", get_blob)
#cv2.imshow("Blob1", blob)
cv2.waitKey(0)

