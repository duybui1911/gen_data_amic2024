import os
import os.path as osp
from glob import glob
from tqdm import tqdm

import cv2
import utils

out = "pre_labels"
labels = glob("tools/data_yolov8/train/labels/*")
images = [
    osp.join("tools/data_yolov8/train/images", 
             osp.basename(name).replace(".txt", ".jpg")) for name in labels]
os.makedirs(out, exist_ok=True)

for (l_path, im_path) in tqdm(zip(labels, images)):
    im_path = im_path.replace("\\", "/")
    image = cv2.imread(im_path)
    h, w = image.shape[:2]
    boxes = utils.load_yolo_boxes(l_path, (w, h), True)

    for box_id, box in enumerate(boxes):
        cls_name, x1, y1, x2, y2 = box
        os.makedirs(f"{out}/{cls_name}", exist_ok=True)
        box_im = image[y1:y2, x1:x2]
        box_name = osp.basename(l_path) + f"_box_{box_id}.jpg"

        cv2.imwrite(
            osp.join(f"{out}/{cls_name}", box_name),
            box_im
        )