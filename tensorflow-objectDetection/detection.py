import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import math

# Load models
model2 = hub.load("./mobilenet-v2-tensorflow1-openimages-v4-ssd-mobilenet-v2-v1").signatures["default"]
model3 = hub.load("./faster-rcnn-inception-resnet-v2-tensorflow1-faster-rcnn-openimages-v4-inception-resnet-v2-v1").signatures["default"]

colorcodes = {}

def drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color):
    im_height, im_width, _ = image.shape
    left, top, right, bottom = int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)
    cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=4)
    FONT_SCALE = 5e-3
    THICKNESS_SCALE = 4e-3
    width = right - left
    height = bottom - top
    TEXT_Y_OFFSET_SCALE = 1e-2
    cv2.rectangle(image, (left, top - int(height * 6e-2)), (right, top), color=color, thickness=-1)
    cv2.putText(image, namewithscore, (left, top - int(height * TEXT_Y_OFFSET_SCALE)),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=min(width, height) * FONT_SCALE,
                thickness=math.ceil(min(width, height) * THICKNESS_SCALE),
                color=(255, 255, 255))

def draw(image, boxes, classnames, scores):
    boxesidx = tf.image.non_max_suppression(boxes, scores, max_output_size=4, iou_threshold=0.5, score_threshold=0.1)
    for i in boxesidx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("ascii")
        if classname in colorcodes:
            color = colorcodes[classname]
        else:
            color = (random.randrange(0, 255, 30), random.randrange(0, 255, 25), random.randrange(0, 255, 50))
            colorcodes[classname] = color
        namewithscore = "{}:{}".format(classname, int(100 * scores[i]))
        drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color)
    return image

def process_image(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (800, 600))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    converted_img = tf.image.convert_image_dtype(image_rgb, tf.float32)[tf.newaxis, ...]
    detection = model(converted_img)
    result = {key: value.numpy() for key, value in detection.items()}
    image_with_boxes = draw(image, result['detection_boxes'], result['detection_class_entities'], result['detection_scores'])
    return image_with_boxes

# Process and display/save images
image_with_boxes = process_image("mandog.jpg", model2)
#image_with_boxes = cv2.resize(image_with_boxes, (0, 0), fx=0.1, fy=0.1)
cv2.imwrite("detectedmandog.jpg", image_with_boxes)
cv2.waitKey(0)

#image_with_boxes = process_image("image4.jpg", model3)
#cv2.imwrite("detected4.jpg", image_with_boxes)

#image_with_boxes = process_image("a.jpg", model3)
#cv2.imwrite("detecteda.jpg", image_with_boxes)

#image_with_boxes = process_image("image3.jpg", model3)
#cv2.imwrite("detected31.jpg", image_with_boxes)
cv2.waitKey(0)
