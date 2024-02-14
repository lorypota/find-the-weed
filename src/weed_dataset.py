import json
import os
import PIL
from PIL import Image
from torchvision import datasets


class DividedWeedDataset(datasets.ImageFolder):
    def __init__(self, ann_file, root, transform=None):
        self.root = root
        self.transform = transform
        with open(os.path.join(root, ann_file)) as file:
            self.json = json.load(file)

    def __getitem__(self, index):
        # considering img_id as passed index
        img_path = os.path.join(self.root,
                                self.json["images"][index]["file_name"])

        anns_for_img = []
        for jsonObj in self.json["annotations"]:
            if jsonObj["image_id"] == index:
                anns_for_img.append(jsonObj)

        image = PIL.Image.open(img_path)

        # print(anns_for_img)

        sub_images = []
        sub_annotations = []  # Store annotations for each sub-image
        for i in range(5):
            for j in range(5):
                x_start = j * 128
                y_start = i * 128
                x_end = x_start + 128
                y_end = y_start + 128

                sub_image = image.crop((x_start, y_start, x_end, y_end))

                if self.transform:
                    sub_image = self.transform(sub_image)
                sub_images.append(sub_image)

                # Scale and filter annotations for this sub-image
                filtered_anns = self.filter_and_scale_anns(anns_for_img, x_start, y_start, 128)
                sub_annotations.append(filtered_anns)

        return sub_images, sub_annotations

    def filter_and_scale_anns(self, annotations, x_start, y_start, size):
        filtered_anns = []
        for ann in annotations:
            bbox = ann['bbox']  # Original bounding box [x, y, width, height]

            # Check if bounding box intersects with sub-image
            if self.bbox_intersects(bbox, x_start, y_start, size):
                # Adjust bounding box coordinates relative to sub-image
                new_x = max(0, bbox[0] - x_start)
                new_y = max(0, bbox[1] - y_start)

                # Adjust width and height
                new_width = min(size, new_x + bbox[2])
                new_height = min(size, new_y + bbox[3])

                # Update annotation
                new_bbox = [new_x, new_y, new_width, new_height]
                filtered_anns.append(new_bbox)

        return filtered_anns

    def bbox_intersects(self, bbox, x_start, y_start, size):
        [x, y, width, height] = bbox
        x_end = x_start + size
        y_end = y_start + size
        bbox_x_end = x + width
        bbox_y_end = y + height
        return (x < x_end and bbox_x_end > x_start and
                y < y_end and bbox_y_end > y_start)
