import os
import json
import argparse
from pycocotools.coco import COCO

class Extractor():
    def __init__(self, annotation_path, images_path, output_path) -> None:        
        self.annotation_path = annotation_path
        self.images_path = images_path
        self.output_path = output_path

        self.annotation = COCO(self.annotation_path)

    def check_folders(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_imgs_names(self):
        imgs_ids = self.annotation.getImgIds()
        imgs = self.annotation.loadImgs(imgs_ids)
        imgs = [img['file_name'] for img in imgs]
        return imgs

    def extract(self):
        self.check_folders(self.output_path)
        imgs = self.get_imgs_names()
        print(f"Extracting {len(imgs)} images")
        for img in imgs:
            img_path = os.path.join(self.images_path, img)
            os.system(f"cp {img_path} {self.output_path}")


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', help='Path to the user annotation of images')
    parser.add_argument('--images_path', help='Path to the user annotation of images')
    parser.add_argument('--output_path', help='Path to the user annotation of images')
    args = parser.parse_args()

    extractor = Extractor(args.annotation_path, args.images_path, args.output_path)
    extractor.extract()