import cv2
import numpy as np

cap = cv2.VideoCapture(0)

classFile = 'coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

while True:
    _, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)


# pip install wget
#
# python -c "import wget; ; wget.download('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'); wget.download('https://pjreddie.com/media/files/yolov3.weights');"
