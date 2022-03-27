import cv2
import numpy as np

# Configs
weights = 'yolo/yolov3.weights'
config = 'yolo/yolov3.cfg'
classes = 'yolo/coco.names'

image = 'res/pizza.jpeg'
image = cv2.imread(image)
width = image.shape[0]
height = image.shape[1]
scale = 0.00392
names = open(classes).read().splitlines()
net = cv2.dnn.readNet(weights, config)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), True, crop=False)
get_blob = blob.reshape(blob.shape[2], blob.shape[3], blob.shape[1])
cv2.imshow("Blob", get_blob)
cv2.imshow("Original", image)
net.setInput(blob)


# Task 10
def get_output_layers(net):
    layer_names = net.getLayerNames()
    print(net.getUnconnectedOutLayers())
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    return output_layers


# Task 11
outs = net.forward(get_output_layers(net))
class_ids = []
confidences = []
boxes = []
score_threshold = 0.5
nms_threshold = 0.4


# Task 16-18
def draw_prediction(img, class_id, confidence, x, y, x1, y1):
    label = str(names[class_id])
    cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


# Task 12
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
print(indices)

for i in indices:
    i = i[0]
    box = boxes[i]
    print(box)
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

cv2.imshow("Prediction", image)
cv2.waitKey(0)