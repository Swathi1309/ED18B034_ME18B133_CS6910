import cv2
import numpy as np
from collections import OrderedDict
import pyttsx3


# Loading the yolo model using a CV backend
def load_yolo():
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def display_blob(blob):
    '''
        Three images each for RED, GREEN, BLUE channel
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


# Function to detect objects
def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


# Getting box dimensions
def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


# Drawing bounding boxes and their labels
def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.7)
    font = cv2.FONT_HERSHEY_SIMPLEX
    final_boxes=[]
    final_labels = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            final_boxes.append([x, x+w, w*h])
            label = str(classes[class_ids[i]])
            final_labels.append(label)
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 3)
    cv2.imshow("Image", img)
    return final_boxes, final_labels


# Deciding what objects to prompt
def location_map(final_boxes, final_labels):
    left = []
    centre = []
    right = []
    for i in range(len(final_boxes)):
        if (final_boxes[i][0] < 50) and (final_boxes[i][2] > 5000) and (final_boxes[i][1] < 200):
            left.append(final_labels[i])
        elif (final_boxes[i][1] > 1440-50) and (final_boxes[i][2] > 5000) and (final_boxes[i][1] < 200):
            right.append(final_labels[i])
        elif (final_boxes[i][2] > 30000):
            centre.append(final_labels[i])
    return left, centre, right


# Initiation of engine for speech
engine = pyttsx3.init()


# TTS function
def text_to_speech(direction, class_img, prev_text):
    class_img = list(OrderedDict.fromkeys(class_img))
    separator = ' and '
    if len(class_img) == 0:
        return prev_text
    elif direction == 'left':
        text = 'There is a ' + separator.join(class_img) + ' to your left'
    elif direction == 'right':
        text = 'There is a ' + separator.join(class_img) + ' to your right'
    else:
        text = 'There is a ' + separator.join(class_img) + ' in front of you'
    if prev_text == text:
        return prev_text

    prev_text = text
    engine.say(text)
    engine.runAndWait()
    return prev_text


# Final function
def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    currentframe = 1
    prev_prev_left = prev_prev_right = prev_prev_centre = 'lalala'

    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        final_boxes, final_labels= draw_labels(boxes, confs, colors, class_ids, classes, frame)
        left_classes, centre_classes, right_classes = location_map(final_boxes, final_labels)
        # print(currentframe)
        if currentframe % 5 == 0:
            prev_left = text_to_speech('left', left_classes, prev_prev_left)
            prev_right = text_to_speech('right', right_classes, prev_prev_right)
            prev_centre = text_to_speech('centre', centre_classes, prev_prev_centre)
            prev_prev_left = prev_left
            prev_prev_right = prev_right
            prev_prev_centre = prev_centre

        currentframe += 1
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


start_video('Final_video.mp4')
