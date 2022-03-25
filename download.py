import wget
import ssl
import os
import sys

def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


if not os.path.exists('res'):
    os.makedirs('res', exist_ok=False)
os.chdir('res')



if not os.path.exists('yolo'):
    os.makedirs('yolo', exist_ok=False)
os.chdir('yolo')

ssl._create_default_https_context = ssl._create_unverified_context

# Names
wget.download('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names', bar=bar_progress)

# CFG
wget.download('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg', bar=bar_progress)

# Weights
wget.download('https://pjreddie.com/media/files/yolov3.weights', bar=bar_progress)
