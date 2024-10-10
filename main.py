import os
import argparse
from pycocotools.coco import COCO
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import numpy as np
import random
import json

seed = 0
random.seed(seed)
np.random.seed(seed)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')


class DataFilter(QtWidgets.QWidget):
    def __init__(self, images_path, annotation_path, output_path) -> None:
        super().__init__()
        self.images_path = images_path
        self.annotation_path = annotation_path
        self.output_path = output_path

        self.annotation = None
        self.new_annotation = {}
        self.images_paths = []
        self.current_image_index = 1668
        self.total_images = 0

        self.colors_count = 0
        self.colors = []

        self.app = None

        self.check_folders()
        self.read_annotation()
        self.load_app()
        self.setup()

        self.init_ui()
        self.show_image()

        self.stop_app()

    def init_ui(self):
        self.setWindowTitle('PiVAT')

        # Layout for the window
        self.layout = QtWidgets.QHBoxLayout()

        # Control images
        self.image_layout = QtWidgets.QHBoxLayout()
        self.original_image_label = QtWidgets.QLabel()
        self.masked_image_label = QtWidgets.QLabel()

        self.image_layout.addWidget(self.original_image_label)
        self.image_layout.addWidget(self.masked_image_label)

        self.layout.addLayout(self.image_layout)

        # Control buttons
        self.button_layout = QtWidgets.QVBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(20)

        self.prev_button = QtWidgets.QPushButton("<- Previous")
        self.prev_button.clicked.connect(self.show_previous_image)
        self.prev_button.setFixedSize(120, 30)
        self.button_layout.addWidget(self.prev_button)

        self.next_button = QtWidgets.QPushButton("Next ->")
        self.next_button.clicked.connect(self.show_next_image)
        self.next_button.setFixedSize(120, 30)
        self.button_layout.addWidget(self.next_button)

        self.save_next_button = QtWidgets.QPushButton("Save and next")
        self.save_next_button.clicked.connect(self.save_image)
        self.save_next_button.setFixedSize(120, 30)
        self.button_layout.addWidget(self.save_next_button)

        self.save_annotation_button = QtWidgets.QPushButton("Save annotation")
        self.save_annotation_button.clicked.connect(self.save_annotation)
        self.save_annotation_button.setFixedSize(120, 30)
        self.button_layout.addWidget(self.save_annotation_button)

        self.stop_button = QtWidgets.QPushButton("Stop filtering")
        self.stop_button.clicked.connect(self.stop_app)
        self.stop_button.setFixedSize(120, 30)
        self.button_layout.addWidget(self.stop_button)

        self.total_image_label = QtWidgets.QLabel(f"Images count: {self.total_images}")
        self.button_layout.addWidget(self.total_image_label)

        self.image_id_label = QtWidgets.QLabel(f"Current image id: {self.current_image_index}")
        self.button_layout.addWidget(self.image_id_label)

        self.done_label = QtWidgets.QLabel(f"")
        self.button_layout.addWidget(self.done_label)

        self.save_label = QtWidgets.QLabel(f"")
        self.button_layout.addWidget(self.save_label)

        # Connect widgets to the layout
        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)
        self.show()

    def show_image(self):
        image_id = self.current_image_index  # Используем индекс как ID
        self.image_info = self.annotation.loadImgs(image_id)

        self.image_path = os.path.join(self.images_path, self.image_info[0]['file_name'])
        self.image = Image.open(self.image_path).convert("RGB")

        # Open original image
        original_image = Image.open(self.image_path).convert("RGB")
        pixmap = self.display_image(original_image)
        self.original_image_label.setPixmap(pixmap)
        
        annotation_ids = self.annotation.getAnnIds(imgIds=image_id)
        annotations = self.annotation.loadAnns(annotation_ids)

        draw = ImageDraw.Draw(self.image)
        for ann in annotations:
            if ann["segmentation"]:
                mask = self.annotation.annToMask(ann)
            else:
                bbox = ann['bbox']  
                x, y, w, h = map(int, bbox)
                img_height, img_width = self.image_info[0]['height'], self.image_info[0]['width']
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                mask[y:y+h, x:x+w] = 1

            color = self.colors[ann["category_id"]]
            text = self.annotation.loadCats(ann["category_id"])[0]["name"]

            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j] == 1:
                        self.image.putpixel((j, i), color)

            draw.text((ann["bbox"][0], ann["bbox"][1]), text, fill=color, font=self.font)

        pixmap = self.display_image(self.image)
        self.masked_image_label.setPixmap(pixmap)

        self.image_id_label.setText(f"Current image id: {self.current_image_index}")

    def display_image(self, image):
        """Display the original image on the left side."""
        image = image.convert("RGBA")
        data = image.tobytes("raw", "RGBA")
        qim = QtGui.QImage(data, image.width, image.height, QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(qim)

        return pixmap

    def show_next_image(self):
        if self.current_image_index < self.total_images - 1:
            self.current_image_index += 1
        else:
            # self.current_image_index = 0
            self.done_label.setText(f"YOU WATCHED ALL IMAGES!!!!!")
        self.show_image()

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
        else:
            self.current_image_index = self.total_images - 1
        self.show_image()

    def save_image(self):
        if self.current_image_index < self.total_images - 1:
            image_id = self.current_image_index
            image_info = self.annotation.loadImgs(image_id)

            image_path = os.path.join(self.images_path, image_info[0]['file_name'])
            image = Image.open(image_path).convert("RGB")

            annotation_ids = self.annotation.getAnnIds(imgIds=image_id)
            annotations = self.annotation.loadAnns(annotation_ids)

            categories_ids = {ann["category_id"] for ann in annotations}
            categories = self.annotation.loadCats(categories_ids)

            if image_info not in self.new_annotation["images"]:
                self.new_annotation["images"].append(image_info[0])

            for ann in annotations:
                if ann not in self.new_annotation["annotations"]:
                    self.new_annotation["annotations"].append(ann)

            for category in categories:
                if category not in self.new_annotation["categories"]:
                    self.new_annotation["categories"].append(category)
        
            image.save(os.path.join(self.output_path, image_info[0]['file_name']))

        # Next image
        self.show_next_image()

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

        # only images files
        self.images_paths = [f for f in os.listdir(self.images_path) 
                             if f.lower().endswith(IMAGE_EXTENSIONS)]
        self.total_images = len(self.images_paths)
        print(f"Total images: {self.total_images}")

    def fix_idxs(self):
        for idx_cat, cat in enumerate(self.new_annotation["categories"]):
            old_id_cat = cat["id"]
            cat["id"] = idx_cat
            
            for idx_omg, img in enumerate(self.new_annotation["images"]):
                old_id_img = img["id"]
                img["id"] = idx_omg

                for ann in self.new_annotation["annotations"]:
                    if ann["category_id"] == old_id_cat:
                        ann["category_id"] = idx_cat
                    if ann["image_id"] == old_id_img:
                        ann["image_id"] = idx_omg

    def save_annotation(self):
        self.fix_idxs()
        with open(os.path.join(self.output_path, "annotations.json"), "w") as f:
            json.dump(self.new_annotation, f)

        self.save_label.setText(f"Annotations saved")

    def setup(self):
        self.colors_count = len(self.annotation.dataset["categories"])
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(self.colors_count)]
        self.font = ImageFont.load_default(size=15)

    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == QtCore.Qt.Key_D:
            self.show_next_image()
        elif event.key() == QtCore.Qt.Key_A:
            self.show_previous_image()
        elif event.key() == QtCore.Qt.Key_S:
            self.save_image()
        elif event.key() == QtCore.Qt.Key_W:
            self.save_annotation()
        elif event.key() == QtCore.Qt.Key_Q:
            self.stop_app()

    def load_app(self):
        self.app = QtWidgets.QApplication(sys.argv)

    def stop_app(self):
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    filter = DataFilter(args.images_path, args.annotation_path, args.output_path)
    sys.exit(app.exec_())