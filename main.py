import cv2
import numpy as np

# Configs

confing = 'yolo/yolov3.cfg'
weights = 'yolo/yolov3.weights'
classes = 'yolo/coco.names'
names = open(classes).read().splitlines()

image = cv2.imread('res/plane.jpeg')
width = image.shape[0]
height = image.shape[1]

print("W: ", width)
print("H: ", height)

# 1/255
scale = 0.00392
net = cv2.dnn.readNet(weights, confing)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), True, crop=False)
get_blob = blob.reshape(blob.shape[2], blob.shape[3], blob.shape[1])

cv2.imshow("Blob", get_blob)
cv2.imshow("Img", image)
net.setInput(blob)
cv2.waitKey(0)


