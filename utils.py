import os
from typing import Tuple, List, Dict

import cv2
import random


def convert_yolo_to_xywh(boxes, img_width, img_height, return_cls=False):
    converted_boxes = []
    for box in boxes:
        box = [float(item) for item in box]
        class_id, x_center, y_center, width, height = box

        # Denormalize values
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Calculate top-left corner
        x = x_center - (width / 2)
        y = y_center - (height / 2)

        x, y, width, height = int(x), int(y), int(width), int(height)
        # Append converted box
        if return_cls:
            converted_boxes.append([int(class_id), x, y, x+width, y+height])
        else:
            converted_boxes.append([x, y, x+width, y+height])

    return converted_boxes


def load_yolo_boxes(file_path, image_shape_wh: Tuple, return_cls=False):
    if not os.path.isfile(file_path):
        raise "Label file not exists."

    with open(file_path, "r") as f:
        boxes = f.readlines()
    boxes = [item.strip().split(" ") for item in boxes]
    return convert_yolo_to_xywh(boxes, image_shape_wh[0], image_shape_wh[1], return_cls)


def write_yolo_labels(boxes, path):
    with open(path, "w") as f:
        for line in boxes:
            class_id, x_center, y_center, width, height = line
            f.write(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def group_labels(boxes, threshold=30, dir: int = 0):
    grouped_boxes = []

    if dir in [0, 3]:
        boxes_with_centers = [(box, (box[1] + box[3]) / 2) for box in boxes]
    elif dir == 1:
        boxes_with_centers = [(box, (box[0] + box[2]) / 2) for box in boxes]
    elif dir == 2:
        boxes = sorted(boxes, key=lambda x: x[1], reverse=False)
        grouped_boxes.append([[boxes[0], -1000]])
        # grouped_boxes.append([boxes[1], boxes[2]])
        boxes_with_centers = [(box, (box[0] + box[2]) / 2)
                              for box in boxes[1:]]

    boxes_with_centers.sort(key=lambda x: x[1])
    for box, center_y in boxes_with_centers:
        added_to_group = False
        for group in grouped_boxes:
            if abs(center_y - group[0][1]) <= threshold:
                group.append((box, center_y))
                added_to_group = True
                break
        if not added_to_group:
            grouped_boxes.append([(box, center_y)])

    return_group = []
    if dir == 3:
        for group in grouped_boxes:
            gr_boxes = [box for box, _ in group]
            gr_boxes = sorted(gr_boxes, key=lambda x: x[0], reverse=False)
            return_group.append(gr_boxes[:2])
            return_group.append(gr_boxes[2:])

    else:
        return_group = [[box for box, _ in group] for group in grouped_boxes]
    return return_group


def clean_image(image: cv2.Mat, boxes: List, base_label_class1: List):
    padding = 2
    for box in boxes:
        class_id, x1, y1, x2, y2 = box
        if int(class_id) == 1:
            continue
        box_empty_path = random.choice(base_label_class1)
        box_empty_image = cv2.imread(box_empty_path)
        box_empty_image[box_empty_image < 100] += 100
        box_empty_image = cv2.resize(
            box_empty_image, (x2-x1+padding, y2-y1+padding))
        image[y1-padding//2:y2+padding//2,
              x1 - padding//2:x2+padding//2] = box_empty_image

    return image


def gen_an_image(
    empty_image: cv2.Mat,
    empty_annotations: List,
    base_label_paths: List,
) -> Tuple[cv2.Mat, List]:

    gen_annos = []
    imh, imw = empty_image.shape[:2]
    flag = False

    start_id = 0
    for anno_group in empty_annotations:
        if len(anno_group) < 1:
            continue

        for box in anno_group:
            x1, y1, x2, y2 = box
            box_w = (x2-x1)
            box_h = (y2-y1)
            gen_annos.append(
                [1, (x1 + box_w//2)/imw, (y1 + box_h//2)/imh,
                 box_w/imw, box_h/imh]
            )
        if len(anno_group) == 1:
            random_sign = random.randint(0, 5) < 2
            flag = True
            if not random_sign:
                continue

        anno_value = random.randint(start_id, len(anno_group)-1)
        if anno_value == 0 and flag:
            start_id = 1
        x1, y1, x2, y2 = anno_group[anno_value]
        label_path = random.choice(base_label_paths)

        label_im = cv2.imread(label_path)

        label_im = cv2.resize(label_im, (x2-x1, y2-y1))
        box_anno = empty_image[y1:y2, x1:x2]

        org_ratio = random.uniform(0.3, 0.5)
        box_anno = cv2.addWeighted(
            box_anno, org_ratio, label_im, 1-org_ratio, 0)
        empty_image[y1:y2, x1:x2] = box_anno
        box_w = (x2-x1)
        box_h = (y2-y1)
        update_box = [0, (x1 + box_w//2)/imw,
                      (y1 + box_h//2)/imh, box_w/imw, box_h/imh]
        gen_annos[anno_value - len(anno_group)] = update_box

    return empty_image, gen_annos
