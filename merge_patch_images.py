import os
import os.path as osp
import cv2
import json
from glob import glob

import utils

iters = 2
json_path = "content/b_json4"
input_image_gen = "outputs"
input_origin_image_folder = "tools/org_images"

output_merge_images = "merge_images"

os.makedirs(output_merge_images + "/images", exist_ok=True)
os.makedirs(output_merge_images + "/labels", exist_ok=True)

input_jsons = glob(json_path + "/*")
image_paths = glob(input_image_gen + "/images/*")
image_paths_wo_iters = list(set([path.split("_iter_")[0] for path in image_paths]))

json_map_dict = {}
for json_path in input_jsons:
    json_name = osp.basename(json_path)[:-5]
    json_map_dict[json_name] = []

    for org_impath in image_paths_wo_iters:
        if json_name not in org_impath:
            continue
        json_map_dict[json_name].append(org_impath)

for iter in range(iters):    
    for json_path in input_jsons:
        json_name = osp.basename(json_path)[:-5]
        list_im_path = json_map_dict[json_name]
        json_values = json.load(open(json_path, "r"))

        if len(list_im_path) <= 0:
            continue
        origin_image = cv2.imread(osp.join(input_origin_image_folder, json_name + ".JPG"))
        origin_image = cv2.resize(origin_image, (2255, 3151))

        bigh, bigw = origin_image.shape[:2]
        full_labels = []
        for item in json_values.keys():
            x, y = json_values[item]
            for im_path in list_im_path:
                im_path = im_path.replace("\\", "/")
                label_path = im_path.replace("/images/", "/labels/")
                if item not in im_path:
                    continue
                patch_im = cv2.imread(
                    im_path + f"_iter_{iter}.jpg"
                )
                ph, pw = patch_im.shape[:2]
                patch_boxes = utils.load_yolo_boxes(
                    label_path + f"_iter_{iter}.txt",
                    (pw, ph),
                    return_cls=True
                )
                for _box in patch_boxes:
                    class_id, x1, y1, x2, y2 = _box
                    x1 += x
                    y1 += y
                    x2 += x
                    y2 += y
                    _box_w = (x2 - x1)
                    _box_h = (y2 - y1)
                    x_c = (x1 + x2) / 2
                    y_c = (y1 + y2) / 2

                    x_c /= bigw
                    y_c /= bigh
                    _box_w /= bigw
                    _box_h /= bigh
                    full_labels.append([class_id, x_c, y_c, _box_w, _box_h])
                origin_image[y:y+ph, x:x+pw] = patch_im
            
        utils.write_yolo_labels(
            full_labels, 
            osp.join(output_merge_images + "/labels", json_name + f"_iter_{iter}.txt"))
        cv2.imwrite(
            osp.join(output_merge_images + "/images", json_name + f"_iter_{iter}.jpg"),
            origin_image
        )

        