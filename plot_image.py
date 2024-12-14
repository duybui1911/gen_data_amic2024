import cv2
import utils

def draw_boxes(image, boxes):
    for box in boxes:
        cls_id, x1, y1, x2, y2 = box
        image = cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255 - int(cls_id)*255, int(cls_id)*255, 0),
            1
        )
    return image

image_path = r"merge_images\images\IMG_1584_iter_0.jpg"
label_path = r"merge_images\labels\IMG_1584_iter_0.txt"

image = cv2.imread(image_path)
h, w = image.shape[:2]
boxes = utils.load_yolo_boxes(label_path, (w, h), True)
image = draw_boxes(image, boxes)
cv2.imwrite("image_draw.jpg", image)