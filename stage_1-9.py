import cv2
import numpy as np

# Configs
weights = 'res/yolov3.weights'
config = 'res/yolov3.cfg'
classes = 'res/coco.names'

image = 'res/plane.jpeg'
image = cv2.imread(image)
width = image.shape[0]
height = image.shape[1]
scale = 0.00392
names = open(classes).read().splitlines()
net = cv2.dnn.readNet(weights, config)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), True, crop=False)
get_blob = blob.reshape(blob.shape[2], blob.shape[3], blob.shape[1])
cv2.imshow("Blob", get_blob)
cv2.imshow("image", image)
cv2.waitKey(0)
net.setinput(blob)
