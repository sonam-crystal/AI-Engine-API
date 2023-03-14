import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import service.core.logic.tiny_yolov4.core.utils as utils
from service.core.logic.tiny_yolov4.core.yolov4 import filter_boxes
from service.core.logic.tiny_yolov4.core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def object_detector(img):
    # flags.DEFINE_string('framework', 'tflite', '(tf, tflite, trt')
    # flags.DEFINE_string('weights', './yolov4-tiny-416-fp16.tflite',
    #                     'path to weights file')
    # flags.DEFINE_integer('size', 416, 'resize images to')
    # flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
    # flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
    # flags.DEFINE_string('images', './data/images/Image552.jpg', 'path to input image')
    # flags.DEFINE_string('output', 'result.png', 'path to output folder')
    # flags.DEFINE_float('iou', 0.30, 'iou threshold')
    # flags.DEFINE_float('score', 0.50, 'score threshold')
    # flags.DEFINE_boolean('crop', True, 'crop detections from images')

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = 416
    # image_path = img
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # The cv2.imread() function reads the image as a NumPy array in BGR format by default
    # original_image = cv2.imread(image_path)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = cv2.resize(img, (input_size, input_size))

    # scale the pixel values to the range of [0, 1]
    image_data = image_data / 255.
    # image_data = image_data[np.newaxis, ...].astype(np.float32)

    # images_data variable contains a NumPy array of preprocessed image(s)
    # shape of the array is (1, input_size, input_size, 3)
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    # if FLAGS.framework == 'tflite':
    interpreter = tf.lite.Interpreter(model_path='service/core/logic/tiny_yolov4/yolov4-tiny-416-fp16.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
   
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    # print(pred[0])
    # print(pred[1])

    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
    # print(boxes)
    # print(pred_conf)

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    # print("boxes",boxes)
    # print("score",scores)
    # print("classes",classes)
    # print("valid_detections",valid_detections)

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

    pred_bbox1 = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
    # read in all class names from config
    class_names = {0: 'display_digital', 1: 'barcode', 2: 'serial_number'}
    # print(class_names)
    
    # custom allowed classes (uncomment line below to allow detections for only people)
    all_classes = list(class_names.values())
    allowed_classes = ['display_digital']

    # if crop flag is enabled, crop each detection and return cropped image
    # if FLAGS.crop:
    crops = crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox1, allowed_classes)
    # Assume crops is a numpy array of cropped images returned by crop_objects function
    for crop in crops:
        if crop.shape[0] > 0 and crop.shape[1] > 0:

            if crop.shape[0] > crop.shape[1]:
            # Rotate the image clockwise by 90 degrees
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

            # cv2.imshow('Cropped Image', crop)
            # cv2.waitKey(0)

            image_data2 = cv2.resize(crop, (input_size, input_size))
            # scale the pixel values to the range of [0, 1]
            image_data2 = image_data2 / 255.
            # image_data = image_data[np.newaxis, ...].astype(np.float32)

            # images_data variable contains a NumPy array of preprocessed image(s)
            # shape of the array is (1, input_size, input_size, 3)
            images_data2 = []
            for i in range(1):
                images_data2.append(image_data2)
            images_data2 = np.asarray(images_data2).astype(np.float32)

            # if FLAGS.framework == 'tflite':
            interpreter2 = tf.lite.Interpreter(model_path="service/core/logic/tiny_yolov4/yolov4-tiny-step3-416-fp16.tflite")
            interpreter2.allocate_tensors()
            input_details2 = interpreter2.get_input_details()
            output_details2= interpreter2.get_output_details()
        
            interpreter2.set_tensor(input_details2[0]['index'], images_data2)
            interpreter2.invoke()
            pred2 = [interpreter2.get_tensor(output_details2[i]['index']) for i in range(len(output_details2))]
            # print(pred[0])
            # print(pred[1])

            boxes2, pred_conf2 = filter_boxes(pred2[0], pred2[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            # print(boxes)
            # print(pred_conf)

            boxes2, scores2, classes2, valid_detections2 = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes2, (tf.shape(boxes2)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf2, (tf.shape(pred_conf2)[0], -1, tf.shape(pred_conf2)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.30,
                score_threshold=0.25
            )

            pred_bbox2 = [boxes2.numpy(), scores2.numpy(), classes2.numpy(), valid_detections2.numpy()]
            # print("boxes",boxes)
            # print("score",scores)
            # print("classes",classes)
            # print("valid_detections",valid_detections)

            # Extract the x-coordinates(xmin&xmax) from the boxes
            x_coords_min = pred_bbox2[0][:, :, 1] 
            x_coords_max = pred_bbox2[0][:, :, 3]

            # Calculate midpoint of x-coordinates
            x_coords_mid = (x_coords_min + x_coords_max) / 2

            # create mask for non-zero x-coordinates
            valid_mask = (x_coords_min > 0) | (x_coords_max > 0)  
            # apply mask to x-coordinates
            x_coords_min = x_coords_min[valid_mask]  
            x_coords_max = x_coords_max[valid_mask]
            x_coords_mid = x_coords_mid[valid_mask]
            # print("x_coords",x_coords)

            sort_indices = np.argsort(x_coords_mid.flatten())
            # print("sorted indices", sort_indices)

            sorted_boxes = pred_bbox2[0][valid_mask][sort_indices]
            sorted_scores = pred_bbox2[1][valid_mask][sort_indices]
            sorted_classes = pred_bbox2[2][valid_mask][sort_indices]

            # print(sorted_classes)

            label_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
            8: '8', 9: '9', 10: 'kwh', 11: 'kvah', 12: 'kva', 13: 'kw', 14: 'pf'}

            # create a string to store the final sorted classes i.e, the reading
            sorted_digits = []
            parameter= []

            # loop through each sorted class ID
            for class_id in sorted_classes:
                # if class ID is less than 15, append corresponding label to the sorted_digits list
                if class_id < 10:
                    sorted_digits.append(label_map[class_id])
                # if class ID is between 10 and 14, skip it for now
                elif 10 <= class_id <= 14:
                    pass
                # if class ID is 15, append '.' to the sorted_digits list
                elif class_id == 15:
                    sorted_digits.append('.')
                # otherwise, raise an exception as invalid class ID
                else:
                    raise ValueError('Invalid class ID: {}'.format(class_id))

            # loop through each sorted class ID again
            for class_id in sorted_classes:
                # if class ID is between 10 and 14, append corresponding label to the sorted_digits list
                if 10 <= class_id <= 14:
                    parameter.append(label_map[class_id])

            # calculate average height of all digits
            # calculate average height of digits (0 to 9)
            digit_heights = []
            for i in range(len(sorted_boxes)):
                if sorted_classes[i] < 10:
                    digit_heights.append(sorted_boxes[i][2] - sorted_boxes[i][0])
            avg_digit_height = sum(digit_heights) / len(digit_heights)
            # print(digit_heights)
            # print(avg_digit_height)

            # find small digits (height < 20% of average digit height)
            small_digit_indices = []
            for i in range(len(sorted_boxes)):
                if sorted_classes[i] < 10:
                    digit_height = sorted_boxes[i][2] - sorted_boxes[i][0]
                    if digit_height < 0.8 * avg_digit_height:
                        small_digit_indices.append(i)
            # print(small_digit_indices)

            # check if small digit is at the end of the reading or in the middle
            if len(small_digit_indices) == 1:
                small_digit_index = small_digit_indices[0]
                if small_digit_index == len(sorted_boxes) - 1 or (small_digit_index == len(sorted_boxes) - 2 and sorted_classes[small_digit_index + 1] >= 10):
                    # Check if there is already a decimal point
                    if '.' not in sorted_digits:
                        # Add decimal point before the smallest digit
                        sorted_digits = sorted_digits[:small_digit_index-1] + ['.'] + sorted_digits[small_digit_index-1:]
                else:
                    # Remove the smallest digit
                    sorted_digits = sorted_digits[:small_digit_index] + sorted_digits[small_digit_index+1:]
                    sorted_boxes = np.delete(sorted_boxes, small_digit_index, axis=0)
                    sorted_scores = np.delete(sorted_scores, small_digit_index, axis=0)
                    sorted_classes = np.delete(sorted_classes, small_digit_index, axis=0)

            warning = False
            for score in sorted_scores:
                if score < 0.4:
                    warning = True
                    break

            # print("READING::", sorted_digits)
            # print("PARAMETER::",parameter)

            if warning:
                return {
                    "warning": "Some scores are less than 0.4.",
                    "reading": "".join(sorted_digits),
                    "annotation": "".join(parameter)
                }
            else:
                return {
                    "reading": "".join(sorted_digits),
                    "annotation": "".join(parameter)
                }

   
            # image2 = utils.draw_bbox(crop, pred_bbox2)
            # # image = utils.draw_bbox(image_data*255, pred_bbox)
            # image2 = Image.fromarray(image2.astype(np.uint8))
            # image2.show()
            # image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_BGR2RGB)
            # cv2.imwrite("result.png", image2)

    # cv2.destroyAllWindows()

    # image = utils.draw_bbox(original_image, pred_bbox)
    # image = utils.draw_bbox(image_data*255, pred_bbox)
    # image = Image.fromarray(image.astype(np.uint8))
    # image.show()
    # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # cv2.imwrite(FLAGS.output, image)
    # cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)

# if __name__ == '__main__':
#     try:
#         app.run(main)
#     except SystemExit:
#         pass
