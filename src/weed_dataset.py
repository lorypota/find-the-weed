import json
import os
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes


def bbox_intersects(bbox, x_start, y_start, size):
    [x, y, width, height] = bbox
    x_end = x_start + size
    y_end = y_start + size
    bbox_x_end = x + width
    bbox_y_end = y + height
    return (x <= x_end and bbox_x_end >= x_start and
            y <= y_end and bbox_y_end >= y_start)


def filter_and_scale_anns(annotations, x_start, y_start, size):
    filtered_anns = []
    for ann in annotations:
        bbox = ann['bbox']  # Original bounding box [x, y, width, height]

        # Check if bounding box intersects with sub-image
        if bbox_intersects(bbox, x_start, y_start, size):
            # Adjust bounding box coordinates relative to sub-image
            new_x = max(0.0, float(bbox[0]) - x_start)
            new_y = max(0.0, float(bbox[1]) - y_start)

            # Adjust width and height
            new_width = min(size, float(bbox[0]) - x_start + bbox[2])
            new_height = min(size, float(bbox[1]) - y_start + bbox[3])

            # Update annotation
            new_bbox = [new_x, new_y, new_width, new_height]
            filtered_anns.append(new_bbox)

    return filtered_anns


class DividedWeedDataset(Dataset):
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

        # considering img_id as passed index
        img_path = os.path.join(self.root,
                                self.json["images"][img_index]["file_name"])
        image = PIL.Image.open(img_path)

        anns_for_img = []
        for jsonObj in self.json["annotations"]:
            if jsonObj["image_id"] == img_index:
                anns_for_img.append(jsonObj)

        # print(anns_for_img)

        i, j = sub_img_index // 5, sub_img_index % 5  # Get row, col index of sub-image
        x_start = j * 128
        y_start = i * 128
        x_end = x_start + 128
        y_end = y_start + 128

        sub_image = image.crop((x_start, y_start, x_end, y_end))

        if self.transform:
            sub_image = self.transform(sub_image)

        # Scale and filter annotations for this sub-image
        filtered_anns = filter_and_scale_anns(anns_for_img, x_start, y_start, 128)
        target = {}
        if len(filtered_anns) > 0:
            filtered_anns_tens = torch.tensor(filtered_anns)

            labels = torch.ones((len(filtered_anns),), dtype=torch.int64)
            area = (filtered_anns_tens[:, 3] - filtered_anns_tens[:, 1]) * (
                    filtered_anns_tens[:, 2] - filtered_anns_tens[:, 0])
            iscrowd = torch.zeros((len(filtered_anns),), dtype=torch.int64)

            target["boxes"] = BoundingBoxes(filtered_anns, format="XYXY", canvas_size=(128, 128))
            target["labels"] = labels
            target["image_id"] = (img_index * 25) + sub_img_index
            target["area"] = area
            target["iscrowd"] = iscrowd

        else:
            area = torch.tensor(128 * 128, dtype=torch.float32)
            target["boxes"] = BoundingBoxes(torch.tensor([0, 0, 128, 128], dtype=torch.float32), format="XYXY",
                                            canvas_size=(128, 128))
            target["labels"] = torch.zeros((1,), dtype=torch.int64)
            target["image_id"] = (img_index * 25) + sub_img_index
            target["area"] = area
            target["iscrowd"] = torch.zeros((1,), dtype=torch.int64)

        return sub_image, target
