import json
import os
import PIL
from PIL import Image

import torch
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes

from old_files.weed_dataset import filter_and_scale_anns


class EffDetCOCODataset(CocoDetection):
    def __init__(self, ann_file, root, transform=None):
        self.root = root
        self.transform = transform
        with open(os.path.join(root, ann_file)) as file:
            self.json = json.load(file)

    def __len__(self):
        num_images = len(self.json["images"])
        return num_images * 25

    def __getitem__(self, index):
        img_index = index // 25
        sub_img_index = index % 25

        img_path = os.path.join(self.root,
                                self.json["images"][img_index]["file_name"])
        image = PIL.Image.open(img_path)

        anns_for_img = []
        for jsonObj in self.json["annotations"]:
            if jsonObj["image_id"] == img_index:
                anns_for_img.append(jsonObj)

        i, j = sub_img_index // 5, sub_img_index % 5  # Get row, col index of sub-image
        x_start = j * 128
        y_start = i * 128
        x_end = x_start + 128
        y_end = y_start + 128

        sub_image = image.crop((x_start, y_start, x_end, y_end))

        img = v2.functional.pil_to_tensor(sub_image).to(torch.float32)

        # Scale and filter annotations for this sub-image
        filtered_anns = filter_and_scale_anns(anns_for_img, x_start, y_start, 128)
        if len(filtered_anns) > 0:
            filtered_anns_tens = torch.tensor(filtered_anns, dtype=torch.float32)

            labels = torch.ones((len(filtered_anns),), dtype=torch.int64)

            bboxes = BoundingBoxes(filtered_anns_tens, format="xywh", canvas_size=(128, 128))
        else:
            return img, None

        # transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
        # transformed_img = transformed["image"]
        # transformed_labels = transformed["labels"]


        target = {
            "bboxes": torch.as_tensor(bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "img_size": (128, 128),
            "img_scale": torch.tensor([1.0]),
        }

        return img, target
