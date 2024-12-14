import os
import json
import argparse
import os.path as osp
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
import utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Arguments for AMIC gen data.')
    parser.add_argument('-i', '--input', type=str,
                        default='test/images', help='Input folder with empty image files')
    parser.add_argument('-o', '--output', type=str,
                        default='outputs', help='Save folder')
    parser.add_argument('-n', '--num', type=int,
                        default=10, help='Num image gen')
    parser.add_argument("-l", "--label", type=str,
                        default="test/labels")
    args = parser.parse_args()
    return args


def main(args):
    NUM_IMAGE = args.num
    out_folder = args.output
    out_folder_images = out_folder + "/images"
    out_folder_labels = out_folder + "/labels"

    os.makedirs(out_folder_images, exist_ok=True)
    os.makedirs(out_folder_labels, exist_ok=True)

    base_images_paths = glob(args.input + "/*")
    labels_paths = [
        osp.join(args.label, osp.basename(path).replace(".jpg", ".txt"))
        for path in base_images_paths
    ]

    map_patchs = json.load(open("map_patch_image.json", "r"))
    base_label_paths = glob("pre_labels/0/*")
    base_label_empty_paths = glob("pre_labels/1/*")

    # num_iters = int(np.ceil(NUM_IMAGE / len(base_images_paths)))

    import time
    start = time.time()
    for iter in tqdm(range(NUM_IMAGE)):
        for (empty_im_path, empty_label_path) in zip(base_images_paths, labels_paths):
            empty_image = cv2.imread(empty_im_path)
            h, w = empty_image.shape[:2]
            empty_labels = utils.load_yolo_boxes(
                empty_label_path, (w, h), True)
            empty_image = utils.clean_image(
                empty_image, empty_labels, base_label_empty_paths
            )
            # print(empty_labels)
            empty_labels = [box[1:] for box in empty_labels]

            mode_group = 0
            for k, v in map_patchs.items():
                if k not in osp.basename(empty_im_path):
                    continue
                mode_group = v
                break

            empty_labels = utils.group_labels(empty_labels, 20, mode_group)
            image_processed, labels = utils.gen_an_image(
                empty_image=empty_image,
                empty_annotations=empty_labels,
                base_label_paths=base_label_paths
            )
            name_image = osp.basename(empty_im_path) + f"_iter_{iter}.jpg"
            name_label = osp.basename(empty_im_path) + f"_iter_{iter}.txt"
            
            cv2.imwrite(f"{out_folder_images}/{name_image}", image_processed)
            utils.write_yolo_labels(labels, f"{out_folder_labels}/{name_label}")
            # assert False

    print("TIME GEN DATA: ", time.time() - start)


if __name__ == "__main__":
    args = parse_args()

    main(args=args)
