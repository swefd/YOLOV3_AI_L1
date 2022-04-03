import cv2
import numpy as np

classes = 'yolo/coco.names'
config = 'yolo/yolov3.cfg'
weights = 'yolo/yolov3.weights'

image = cv2.imread('res/plane.jpeg')

width = image.shape[0]
height = image.shape[1]

scale = 0.00392  # = 1/255
names = open(classes).read().splitlines()
net = cv2.dnn.readNet(weights, config)

blob = cv2.dnn.blobFromImage(image, scale, (416, 416), True, False)
get_blob = blob.reshape(blob.shape[2], blob.shape[3], blob.shape[1])

net.setInput(blob)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print(output_layers)
    return output_layers


def drawPrediction(image, class_id, confidence, x, y, x1, y1):
    label = str(names[class_id]) + ' ' + str(confidence * 100)[:2] + "%"
    cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 2)
    cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


outs = net.forward(get_output_layers(net))

class_ids = []
boxes = []
confidences = []
score_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > score_threshold:
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

print("Indices: ")
print(indices)

for i in indices:
    #i = i[0]
    box = boxes[i]
    print("Box")
    print(box)
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    drawPrediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

cv2.imshow("Prediction", image)
cv2.waitKey(0)










