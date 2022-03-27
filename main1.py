import cv2
import numpy as np

classes = 'yolo/coco.names'
config = 'yolo/yolov3.cfg'
weights = 'yolo/yolov3.weights'

image = cv2.imread('res/plane.jpeg')

width = image.shape[0]
height = image.shape[1]
scale = 0.00392  # = 1/255
name = open(classes).read().splitlines()
net = cv2.dnn.readNet(weights, config)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), True, False)
get_blob = blob.reshape(blob.shape[2], blob.shape[3], blob.shape[1])

net.setInput(blob)

cv2.imshow("Blob", get_blob)
cv2.imshow("Image", image)
cv2.waitKey(0)
