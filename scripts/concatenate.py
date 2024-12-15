import json
import argparse
from pycocotools.coco import COCO

class Concatenator():
    new_annotation = None
    first_annotation = None
    second_annotation = None

    def __init__(self, first_annotation_path, second_annotation_path, output_path) -> None:        
        self.output_path = output_path
        self.first_annotation_path = first_annotation_path
        self.second_annotation_path = second_annotation_path

        self.first_annotation = COCO(first_annotation_path)
        self.second_annotation = COCO(second_annotation_path)

    def init_annotation(self):
        return {
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": ""
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "",
                    "url": ""
                }
            ],
            "categories": [],
            "images": [],
            "annotations": []
        }
    
    def save_annotation(self):
        with open(self.output_path, 'w') as f:
            json.dump(self.new_annotation, f)
        print(f"Annotation saved in {self.output_path}")

    def add_cats(self):
        cats_mapping = {}
        start_idx = len(self.new_annotation['categories'])
        cats_to_add = self.second_annotation.loadCats(self.second_annotation.getCatIds())
        for idx, cat in enumerate(cats_to_add):
            if cat not in self.new_annotation['categories']:
                new_idx = start_idx + idx + 1
                cats_mapping[cat['id']] = new_idx
                cat['id'] = new_idx
                self.new_annotation['categories'].append(cat)
        return cats_mapping
    
    def add_imgs(self):
        imgs_mapping = {}
        start_idx = len(self.new_annotation['images'])
        imgs_to_add = self.second_annotation.loadImgs(self.second_annotation.getImgIds())
        for idx, img in enumerate(imgs_to_add):
            if img not in self.new_annotation['images']:
                new_idx = start_idx + idx + 1
                imgs_mapping[img['id']] = new_idx
                img['id'] = new_idx
                self.new_annotation['images'].append(img)
        return imgs_mapping
    
    def add_anns(self, imgs_mapping):
        start_idx = len(self.new_annotation['annotations'])
        anns_to_add = self.second_annotation.loadAnns(self.second_annotation.getAnnIds())
        for idx, ann in enumerate(anns_to_add):
            ann['id'] = start_idx + idx + 1
            # ann['category_id'] = cats_mapping[ann['category_id']]
            ann['image_id'] = imgs_mapping[ann['image_id']]
            self.new_annotation['annotations'].append(ann)

    def main(self):
        self.new_annotation = self.init_annotation()
        self.new_annotation['categories'] = self.first_annotation.dataset['categories']
        self.new_annotation['images'] = self.first_annotation.dataset['images']
        self.new_annotation['annotations'] = self.first_annotation.dataset['annotations']

        # cats_mapping = self.add_cats()
        imgs_mapping = self.add_imgs()
        self.add_anns(imgs_mapping)

        self.save_annotation()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_annotation_path', help='Path to the user annotation of images')
    parser.add_argument('--second_annotation_path', help='Path to the user annotation of images')
    parser.add_argument('--new_annotation_path', help='Path to the user annotation of images')
    args = parser.parse_args()

    concatenator = Concatenator(args.first_annotation_path, args.second_annotation_path, args.new_annotation_path)
    concatenator.main()