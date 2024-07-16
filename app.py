import os
import urllib.request
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile

def download_weights():
    """
    Download YOLOv3 weights file if it does not exist.
    """
    weights_path = "yolov3.weights"
    if not os.path.exists(weights_path):
        url = "https://pjreddie.com/media/files/yolov3.weights"
        urllib.request.urlretrieve(url, weights_path)
        print("Weights file downloaded.")
    else:
        print("Weights file already exists.")

def load_yolo():
    """
    Load YOLOv3 model and configuration files.
    """
    download_weights()
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    
    if isinstance(unconnected_out_layers[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, output_layers, colors

def detect_objects(img, net, output_layers):
    """
    Detect objects in the image using YOLOv3.
    """
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs

def get_box_dimensions(outs, height, width):
    """
    Get bounding box dimensions for detected objects.
    """
    boxes = []
    confs = []
    class_ids = []
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
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img):
    """
    Draw labels and bounding boxes on the image.
    """
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    person_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                person_count += 1
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.putText(img, f'Persons: {person_count}', (10, 20), font, 2, (0, 255, 0), 2)
    return img

# Initialize YOLO
net, classes, output_layers, colors = load_yolo()

st.title("PY'S Human Detection App")
st.write("Upload an image, video, or use the webcam to detect people.")

option = st.selectbox(
    'Choose input source',
    ('Upload Image', 'Upload Video', 'Use Webcam'))

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key='image_uploader')
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image.convert('RGB'))
        height, width, channels = image.shape
        outs = detect_objects(image, net, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outs, height, width)
        result_image = draw_labels(boxes, confs, colors, class_ids, classes, image)
        st.image(result_image, caption='Processed Image', use_column_width=True)

elif option == 'Upload Video':
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"], key='video_uploader')
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        video = cv2.VideoCapture(video_path)
        stframe = st.empty()
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            height, width, channels = frame.shape
            outs = detect_objects(frame, net, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outs, height, width)
            result_frame = draw_labels(boxes, confs, colors, class_ids, classes, frame)
            stframe.image(result_frame, channels='BGR', use_column_width=True)
        video.release()

elif option == 'Use Webcam':
    if st.button('Start Webcam', key='start_webcam_button'):
        video = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = video.read()
            if not ret:
                break
            height, width, channels = frame.shape
            outs = detect_objects(frame, net, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outs, height, width)
            result_frame = draw_labels(boxes, confs, colors, class_ids, classes, frame)
            stframe.image(result_frame, channels='BGR', use_column_width=True)
            if st.button('Stop Webcam', key='stop_webcam_button'):
                break
        video.release()

st.write("Developed with OpenCV, YOLOv3, and Streamlit.")
