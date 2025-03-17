from linecache import cache

import tensorflow as tf
import numpy as np

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
        boxes -- tensor of shape (19, 19, 5, 4)
        box_confidence -- tensor of shape (19, 19, 5, 1)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    box_scores = box_confidence * box_class_probs

    box_classes = tf.math.argmax(box_scores, axis=-1) # find the index of the class with the highest score
    box_classes_scores = tf.math.reduce_max(box_scores, axis=-1) # find the max score in a box

    filtering_mask = box_classes_scores >= threshold

    # filter to keep only boxes that met the condition
    scores = tf.boolean_mask(box_classes_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)


    return scores, boxes, classes

def iou(boxes1, boxes2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """

    (box1_x1, box1_y1, box1_x2, box1_y2) = boxes1
    (box2_x1, box2_y1, box2_x2, box2_y2) = boxes2

    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    # Cal the area of the interaction
    inter_with = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_with * inter_height

    # Cal the union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None, ), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    # casting the output from yolo_filter_boxes() to the same type
    boxes = tf.cast(boxes, tf.float32)
    scores = tf.cast(scores, tf.float32)

    nms_indices = []
    class_labels = tf.unique(classes)[0] # get unique classes

    for label in class_labels:
        filtering_mask = classes = label

        # filter boxes taking only class have matching label
        boxes_label = tf.boolean_mask(boxes, filtering_mask)

        # filter scores taking only class have matching label
        scores_label = tf.boolean_mask(scores, filtering_mask)

        if tf.shape(boxes_label)[0] > 0: # check if there is any boxes left
            # use nms built in method to get the list indices corresponding to boxes that be kept
            nms_indices_label = tf.image.non_max_suppression(boxes_label, scores_label, max_boxes = max_boxes, iou_threshold = iou_threshold)

            # get the original indices of the selected boxes and squeeze the size to 1D array
            selected_indices = tf.squeeze(tf.where(filtering_mask), axis=1)

            # Append the resulting boxes into the partial result
            nms_indices.append(tf.gather(selected_indices, nms_indices_label))

        nms_indices = tf.concat(nms_indices, axis = 0)

        scores = tf.gather(scores, nms_indices)
        boxes = tf.gather(boxes, nms_indices)
        classes = tf.gather(classes, nms_indices)

        # Sorted by scores and return max top boxes
        sort_order = tf.argsort(scores, direction='DESCENDING').numpy()
        scores = tf.gather(scores, sort_order[0:max_boxes])
        boxes = tf.gather(boxes, sort_order[0:max_boxes])
        classes = tf.gather(classes, sort_order[0:max_boxes])

        return scores, boxes, classes


def yolo_boxes_to_corners(box_xy, box_wh):
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2], # Y MIN
        box_mins[..., 0:1], # x min
        box_maxes[..., 1:2], # Y max
        box_maxes[..., 0:1], # x max
    ])

def yolo_eval(yolo_outputs, image_shap=(720, 1280), max_boxes=10, score_thresold=.6, iou_threshold=0.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    # Retrieve the output of the yolo model
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs

    # convert boxes box_xy and box_wh to conner coordinates
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # filter the classes with high score
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs,score_thresold)
    # boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = iou_threshold)

    return scores, boxes, classes

