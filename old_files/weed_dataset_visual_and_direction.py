import json
import os
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes


class WeedDatasetEvaluation(Dataset):
    def __init__(self, ann_file, root, transform=None):
        self.root = root
        self.transform = transform
        with open(os.path.join(root, ann_file)) as file:
            self.json = json.load(file)

    def __len__(self):
        num_images = len(self.json["images"])
        return num_images

    def __getitem__(self, index):
        img_path = os.path.join(self.root,
                                self.json["images"][index]["file_name"])
        image = PIL.Image.open(img_path)

        anns_for_img = []
        for jsonObj in self.json["annotations"]:
            if jsonObj["image_id"] == index:
                anns_for_img.append(jsonObj)

        target = {}
        boxes = []
        for ann in anns_for_img:
            boxes.append(ann['bbox'])

        labels = torch.ones((len(boxes),), dtype=torch.int64)
        # xywh = x y width height
        target["boxes"] = BoundingBoxes(boxes, format="xywh", canvas_size=(640, 640), dtype=torch.float32)
        target["labels"] = labels
        target["imageId"] = torch.tensor(index, dtype=torch.int64)

        new_image = image
        new_target = target

        if self.transform:
            new_image, new_target = self.transform(image, target)

        return new_image, new_target
