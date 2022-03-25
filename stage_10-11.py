import cv2
import numpy as np

# Configs
weights = 'yolo/yolov3.weights'
config = 'yolo/yolov3.cfg'
classes = 'yolo/coco.names'

image = 'res/plane.jpeg'
image = cv2.imread(image)
width = image.shape[0]
height = image.shape[1]
scale = 0.00392
names = open(classes).read().splitlines()
net = cv2.dnn.readNet(weights, config)


# Task 10
def get_output_layers(net):
    layer_names = net.getLayerNames()
    print(net.getUnconnectedOutLayers())
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


# Task 11
outs = net.forward(get_output_layers(net))
class_ids = []
confidences = []
score_threshold = 0.5
nms_threshold = 0.5

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), True, crop=False)
get_blob = blob.reshape(blob.shape[2], blob.shape[3], blob.shape[1])
cv2.imshow("Blob", get_blob)
cv2.imshow("image", image)
cv2.waitKey(0)
net.setinput(blob)
