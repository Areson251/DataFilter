import json
import argparse
from pycocotools.coco import COCO

ALLOWED_CATEGORIES = [
    "pothole",
]

class Fixer():
    annotation = None
    new_annotation = None

    def __init__(self, annotation_path, output_path) -> None:        
        self.annotation_path = annotation_path
        self.output_path = output_path

        self.annotation = COCO(self.annotation_path)
    
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

    def main(self):
        self.new_annotation = self.init_annotation()

        cats_to_add_ids = self.annotation.getCatIds(catNms=ALLOWED_CATEGORIES)
        imgs_ids = self.annotation.getImgIds(catIds=cats_to_add_ids)
        anns_ids = self.annotation.getAnnIds(catIds=cats_to_add_ids)

        true_cat = {
        "id": 1,
        "name": "pothole",
        "supercategory": "",
        } 

        imgs_to_add = self.annotation.loadImgs(imgs_ids)
        img_ids_mapping = {}
        for idx, img in enumerate(imgs_to_add):
            img_ids_mapping[img['id']] = idx + 1
            img['id'] = idx + 1

        anns_to_add = self.annotation.loadAnns(anns_ids)
        for idx, ann in enumerate(anns_to_add):
            if ann['segmentation'] != []:
                segm = self.annotation.annToRLE(ann)
                if isinstance(segm['counts'], bytes):
                    segm['counts'] = segm['counts'].decode('utf-8')
            else:
                segm = []
            ann['id'] = idx + 1
            ann['image_id'] = img_ids_mapping[ann['image_id']]
            ann["category_id"] = 1
            ann["segmentation"] = segm

        self.new_annotation["categories"].append(true_cat)
        self.new_annotation["images"] = imgs_to_add
        self.new_annotation["annotations"] = anns_to_add

        self.save_annotation()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', help='Path to the user annotation of images')
    parser.add_argument('--output_path', help='Path to the user annotation of images')
    args = parser.parse_args()

    concatenator = Fixer(args.annotation_path, args.output_path)
    concatenator.main()