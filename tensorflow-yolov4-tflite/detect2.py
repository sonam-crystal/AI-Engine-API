import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.functions import *
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tflite', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416-fp16.tflite',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/Image50190.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    image_path = FLAGS.image

    # The cv2.imread() function reads the image as a NumPy array in BGR format by default
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = cv2.resize(original_image, (input_size, input_size))

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
    interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
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
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )

    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    # print("boxes",boxes)
    # print("score",scores)
    # print("classes",classes)
    # print("valid_detections",valid_detections)

    # Extract the x-coordinates from the boxes
    x_coords = pred_bbox[0][:, :, 1]  
    # create mask for non-zero x-coordinates
    valid_mask = x_coords > 0  
    # apply mask to x-coordinates
    x_coords = x_coords[valid_mask]  
    # print("x_coords",x_coords)

    sort_indices = np.argsort(x_coords.flatten())
    # print("sorted indices", sort_indices)

    sorted_boxes = pred_bbox[0][valid_mask][sort_indices]
    sorted_scores = pred_bbox[1][valid_mask][sort_indices]
    sorted_classes = pred_bbox[2][valid_mask][sort_indices]

    # print(sorted_classes)

    label_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
     8: '8', 9: '9', 10: 'kwh', 11: 'kvah', 12: 'kva', 13: 'kw', 14: 'pf'}

    # create a string to store the final sorted classes i.e, the reading
    sorted_class_names = ''

    # loop through each sorted class ID
    for class_id in sorted_classes:
        # if class ID is less than 15, append corresponding label to the sorted_class_names list
        if class_id < 10:
            sorted_class_names+=label_map[class_id] + ''
        # if class ID is between 10 and 14, skip it for now
        elif 10 <= class_id <= 14:
            pass
        # if class ID is 15, append '.' to the sorted_class_names list
        elif class_id == 15:
            sorted_class_names+='.'
        # otherwise, raise an exception as invalid class ID
        else:
            raise ValueError('Invalid class ID: {}'.format(class_id))

    # loop through each sorted class ID again
    for class_id in sorted_classes:
        # if class ID is between 10 and 14, append corresponding label to the sorted_class_names list
        if 10 <= class_id <= 14:
            sorted_class_names += ' ' + label_map[class_id]

    print("READING::", sorted_class_names)

    image = utils.draw_bbox(original_image, pred_bbox)
    # image = utils.draw_bbox(image_data*255, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(FLAGS.output, image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
