import os
import argparse
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import numpy as np
import random

class DataFilter():
    def __init__(self, images_path, annotation_path, output_path) -> None:
        self.images_path = images_path
        self.annotation_path = annotation_path
        self.output_path = output_path

        self.annotation = None
        self.new_annotation = {}

        self.app = None
        self.window = None

        self.check_folders()
        self.read_annotation()
        self.load_app()

        self.filter()
        self.stop_app()

    def filter(self):
        images_paths = os.listdir(self.images_path)
        print(f"Total images: {len(images_paths)}")

        for id, image_file in enumerate(images_paths):
            image_id = id
            image_path = os.path.join(self.images_path, image_file)
            image = Image.open(image_path).convert("RGB")
            
            annotation_ids = self.annotation.getAnnIds(imgIds=image_id)
            annotations = self.annotation.loadAnns(annotation_ids)

            draw = ImageDraw.Draw(image)
            for ann in annotations:
                mask = self.annotation.annToMask(ann)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random color for each mask

                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i, j] == 1:
                            image.putpixel((j, i), color)

            self.display_image_with_masks(image)

    def check_folders(self):
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(f"Images path {self.images_path} not found.")
        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"Annotation path {self.annotation_path} not found.")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def read_annotation(self):
        self.annotation = COCO(self.annotation_path)
        self.new_annotation["info"] = self.annotation.dataset["info"]
        self.new_annotation["licenses"] = self.annotation.dataset["licenses"]
        self.new_annotation["categories"] = []
        self.new_annotation["images"] = []
        self.new_annotation["annotations"] = []

    def load_app(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QWidget()

    def display_image_with_masks(self, image):
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        qim = QtGui.QImage(data, image.width, image.height, QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(qim)

        label = QtWidgets.QLabel()
        label.setPixmap(pixmap)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        self.window.setLayout(layout)
        self.window.setWindowTitle('Image with Masks')
        self.window.show()

    def stop_app(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    filter = DataFilter(args.images_path, args.annotation_path, args.output_path)