import json
import os
from PIL import Image, ImageDraw
import numpy as np
from pycocotools import mask as mask_util
from datetime import datetime

# Mapping von Klassennamen mit Umlauten zu Klassennamen ohne Umlaute
CLASS_NAME_MAPPING = {
    "Gelb-grün-braunes Blatt": "Gelb-gruen-braunes Blatt",
    "Asymptomatische Trauben": "Asymptomatische Trauben",
    "Symptomatische Trauben": "Symptomatische Trauben",
    "Gelb-grünes Blatt": "Gelb-gruenes Blatt",
    "Abgestorbenes Blatt": "Abgestorbenes Blatt",
    "Gelbes Blatt": "Gelbes Blatt",
    "Gelber Rand": "Gelber Rand"
}

def validate_polygon(polygon, width, height):
    if not polygon or len(polygon) < 3:
        return False
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False
        x, y = point
        if not (0 <= x < width and 0 <= y < height):
            return False
    return True

def create_coco_annotation(output_json, images_folder, annotations_folder, labels_folder):
    coco_output = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    categories = {
        "Gelb-gruen-braunes Blatt": 0,
        "Asymptomatische Trauben": 1,
        "Symptomatische Trauben": 2,
        "Gelb-gruenes Blatt": 3,
        "Abgestorbenes Blatt": 4,
        "Gelbes Blatt": 5,
        "Gelber Rand": 6
    }

    for category, category_id in categories.items():
        coco_output["categories"].append({
            "id": category_id,
            "name": category,
            "supercategory": "none"
        })

    annotation_id = 1
    image_id = 1

    label_files = {os.path.splitext(os.path.basename(f))[0].replace('_label', '') for f in os.listdir(labels_folder)}

    for image_filename in os.listdir(images_folder):
        if not image_filename.endswith(('.jpg', '.png')):
            continue
        image_base_name = os.path.splitext(image_filename)[0]
        image_path = os.path.join(images_folder, image_filename)
        annotation_path = os.path.join(annotations_folder, image_base_name + '.json')
        label_path = os.path.join(labels_folder, image_base_name + '_label.png')

        print(f"Processing image: {image_filename}")

        if not os.path.exists(annotation_path):
            print(f"Skipping image {image_filename} because annotation is missing")
            continue

        if image_base_name not in label_files:
            print(f"Skipping image {image_filename} because label is missing")
            continue

        try:
            image = Image.open(image_path)
            width, height = image.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        coco_output["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": os.path.join(images_folder, image_filename).replace('\\', '/'),
            "license": 1
        })

        with open(annotation_path, 'r', encoding='utf-8') as f:
            try:
                annotation_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for {annotation_path}: {e}")
                continue

            for shape in annotation_data['shapes']:
                label = shape.get('label')
                if label in CLASS_NAME_MAPPING:
                    label = CLASS_NAME_MAPPING[label]

                if label not in categories:
                    print(f"Invalid label '{label}' in {annotation_path}, skipping this shape.")
                    continue

                category_id = categories[label]
                polygon = shape.get('points')

                if not validate_polygon(polygon, width, height):
                    print(f"Invalid polygon in {annotation_path}, skipping this shape.")
                    continue

                polygon_int = [(int(x), int(y)) for x, y in polygon]
                polygon_flat = [coord for point in polygon_int for coord in point]

                try:
                    mask = Image.new('L', (width, height), 0)
                    ImageDraw.Draw(mask).polygon(polygon_int, outline=1, fill=1)
                    mask = np.array(mask, dtype=np.uint8)

                    rle = mask_util.encode(np.asfortranarray(mask))

                    if rle is None:
                        raise ValueError("RLE encoding failed, got None")

                    rle = {k: (v.decode('utf-8') if isinstance(v, bytes) else v) for k, v in rle.items()}

                except Exception as e:
                    print(f"Error processing polygon in {annotation_path}: {e}")
                    continue

                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "area": int(mask_util.area(rle)),
                    "bbox": list(map(int, mask_util.toBbox(rle))),
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, indent=4, ensure_ascii=False)

    print(f"COCO annotations saved to {os.path.abspath(output_json)}")

def add_missing_info_fields(coco_json_path):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    if "info" not in coco_data:
        coco_data["info"] = {}

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    default_info = {
        "year": datetime.now().year,
        "version": "1.0",
        "description": "Description of the dataset",
        "contributor": "Contributor Name",
        "date_created": current_date
    }

    for key, value in default_info.items():
        if key not in coco_data["info"]:
            coco_data["info"][key] = value

    if "licenses" not in coco_data or not coco_data["licenses"]:
        coco_data["licenses"] = [{
            "id": 1,
            "name": "License Name",
            "url": "http://example.com/license"
        }]

    with open(coco_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)

    print("Missing info fields added to COCO JSON.")


def fix_malformed_annotations(coco_json_path):
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for annotation in data['annotations']:
        if 'size": ' in str(annotation):
            if isinstance(annotation['size'], list):
                annotation['size'] = [int(x) for x in annotation['size']]

    with open(coco_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("Fixed malformed annotations.")


def main():
    images_folder = 'data/escasymptoms/images'
    annotations_folder = 'data/escasymptoms/jsons'
    labels_folder = 'data/escasymptoms/labels'
    coco_folder = 'data/escasymptoms/coco'
    output_json = os.path.join(coco_folder, 'coco_annotations.json')

    if not os.path.exists(coco_folder):
        os.makedirs(coco_folder)

    create_coco_annotation(output_json, images_folder, annotations_folder, labels_folder)
    add_missing_info_fields(output_json)
    fix_malformed_annotations(output_json)

if __name__ == "__main__":
    main()
