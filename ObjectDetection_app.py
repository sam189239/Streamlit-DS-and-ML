import torch
import cv2
import time
import numpy as np
import warnings
from PIL import Image
import streamlit as st
import pandas as pd

warnings.simplefilter('ignore')

# video = r"..\data\client_vid_1.mp4"
video = 0

classes = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "hair brush"
]
default_classes = ['person', 'bicycle', 'car', 'motorcycle']

## Setting parameters and variables##
font = cv2.FONT_HERSHEY_SIMPLEX
threshold = 0.5  ###
roi = 0.3  ###
success = True
images = []
FPS = []

@st.cache(persist=True)
def load_model():

    ## Loading model ##
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model

@st.cache
def model_inference(model, frame, COLORS, threshold, roi, classes):

    objects = {}
    ## Detection ##
    start_time  = time.time()
    output = model(frame)      
    results = output.pandas().xyxy[0]
    end_time = time.time()
    duration = end_time - start_time   ###

    ## Region of Interest ##
    height, width= frame.shape[:2]  
    left, right = int(roi * width), int((1-roi) * width)
    mid = int(width / 2)
    ROI_region = [[(int(roi * width),height),(int(roi * width),0),(int((1-roi) * width),0),(int((1-roi) * width),height)]]
    ROI_region2 = [[(int(roi * width),height),(int(roi * width),0),(int((0.5) * width),0),(int((0.5) * width),height)]]
    box_img = frame

    ## Drawing boxes ##
    for result in results.to_numpy():
        confidence = result[4]
        if confidence >= threshold and result[6] in classes:
            x1,y1,x2,y2,label = int(result[0]),int(result[1]),int(result[2]),int(result[3]),result[6]
            area = (x2-x1) * (y2-y1)
            area = int(area/100)
            box_img = frame
            color = COLORS[classes.index(label)] #(0,255,0)
            
            if x2>=left and x1<=right: # Obstacle inside ROI
                if label in objects.keys():
                    objects[label] = objects[label] +1
                else:
                    objects[label] = 1
                box_img = cv2.rectangle(box_img, (x1,y1),(x2,y2),color,2)
                box_img = cv2.putText(box_img,label,(x1-1,y1-1),font,0.5,(255,0,255),2)
                box_img = cv2.putText(box_img,"Conf: "+str(format(confidence,".2f"))+", Area:"+str(area),(x2-2,y1-1),font,0.5,(255,0,255),2)
        else:
            continue
    

    ## ROI Region ##
    box_img = cv2.rectangle(box_img, ROI_region[0][1],ROI_region[0][3],(0,0,0),1)
    box_img = cv2.rectangle(box_img, ROI_region2[0][1],ROI_region2[0][3],(0,0,0),1)
    
    return box_img, duration, objects



DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEMO_IMAGE = "images/sample.jpg"
CLASSES = classes

st.title("Object detection with YOLOv5s")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
confidence_threshold = st.slider(
    "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)

roi_val = st.slider(
    "Region of interest", 0.0, 1.0, roi, 0.05
)

classes_chosen = st.multiselect("Choose Classes", classes, default=default_classes)


COLORS = np.random.uniform(0, 255, size=(len(classes_chosen), 3))

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

model = load_model()
detection, duration, objects = model_inference(model, image, COLORS, confidence_threshold, (roi_val/2), classes_chosen)

st.image(
    detection, caption=f"Processed image", use_column_width=True,
)

st.write("Time taken: " + str(duration))

detected_classes = {'Objects':[], 'Count':[]}
for a in objects.keys():
    detected_classes['Objects'].append(a)
    detected_classes['Count'].append(objects[a])

st.subheader("Detected Classes:")
detected_classes = pd.DataFrame(detected_classes)
st.table(detected_classes)