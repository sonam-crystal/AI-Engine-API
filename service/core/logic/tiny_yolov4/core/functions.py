import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from service.core.logic.tiny_yolov4.core.utils import read_class_names
from service.core.logic.tiny_yolov4.core.config import cfg

# function to count objects, can return total classes or count per class
def count_objects(data, by_class = False, allowed_classes = list(read_class_names(cfg.YOLO.CLASSES).values())):
    boxes, scores, classes, num_objects = data

    #create dictionary to hold count of objects
    counts = dict()

    # if by_class = True then count objects per class
    if by_class:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # loop through total number of objects found
        for i in range(num_objects):
            # grab class index and convert into corresponding class name
            class_index = int(classes[i])
            class_name = class_names[class_index]
            if class_name in allowed_classes:
                counts[class_name] = counts.get(class_name, 0) + 1
            else:
                continue

    # else count total objects found
    else:
        counts['total object'] = num_objects
    
    return counts

# function for cropping each detection and saving as new image
def crop_objects(img, data, allowed_classes):
    boxes, scores, classes, num_objects = data
    class_names = {0: 'display_digital', 1: 'barcode', 2: 'serial_number'}
    cropped_imgs = []
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name == 'display_digital':
            xmin, ymin, xmax, ymax = boxes[i]
            # adjust crop window to stay within image bounds
            h, w, _ = img.shape
            xmin = max(int(xmin) - 5, 0)
            ymin = max(int(ymin) - 5, 0)
            xmax = min(int(xmax) + 5, w)
            ymax = min(int(ymax) + 5, h)
            # crop detection from image
            # print('xmin:', xmin, 'ymin:', ymin, 'xmax:', xmax, 'ymax:', ymax)
            # cropped_img = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
            cropped_img = img[ymin:ymax, xmin:xmax]
            cropped_imgs.append(cropped_img)
    return cropped_imgs
        
# function to run general Tesseract OCR on any detections 
def ocr(img, data):
    boxes, scores, classes, num_objects = data
    class_names = read_class_names(cfg.YOLO.CLASSES)
    for i in range(num_objects):
        # get class name for detection
        class_index = int(classes[i])
        class_name = class_names[class_index]
        # separate coordinates from box
        xmin, ymin, xmax, ymax = boxes[i]
        print(boxes[i])
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        box = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

        box1 = cv2.cvtColor(box, cv2.COLOR_BGR2RGB)

        cv2.imshow('Box Image', box)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # grayscale region within bounding box
        gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)

        cv2.imshow('Box Image', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # threshold the image using Otsus method to preprocess for tesseract
        # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Invert the thresholded image
        thresh = cv2.bitwise_not(thresh)

        cv2.imshow('Box Image', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # perform a median blur to smooth image slightly
        blur = cv2.medianBlur(thresh, 3)

        cv2.imshow('Box Image', blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # resize image to double the original size as tesseract does better with certain text size
        blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        # run tesseract and convert image text to string

        cv2.imshow('Box Image', blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        try:
            text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
            print("Class: {}, Text Extracted: {}".format(class_name, text))
        except: 
            text = None