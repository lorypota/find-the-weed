import os
import torch
import json
import imageio.v3 as iio
from PIL import Image
from torch.utils.data import Dataset


class WeedDataset(Dataset):
    """Weed dataset."""

    def __init__(self, json_file, root_dir, transform=None):
        with open(os.path.join(root_dir, json_file)) as file:
            self.json = json.load(file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.json["images"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.json["images"][idx]["file_name"])
        image = iio.imread(img_name)

        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)

        annotations = []
        for jsonObj in self.json["annotations"]:
            if jsonObj["image_id"] == idx:
                annotations.append(jsonObj)

        sample = {'image': image, 'annotations': annotations}

        boxes = []
        # labels = []
        for obj in sample['annotations']:
            xmin, ymin, w, h = obj['bbox']
            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])  # Create bounding box tensor
            # labels.append(obj['category_id'])       # Extract class label

        # Transform data into proper tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Padding
        num_boxes = len(boxes)
        max_detections = 10 # TODO: change handling of max_detections
        if num_boxes < max_detections:
            boxes = torch.nn.functional.pad(boxes, (0, 0, 0, max_detections - num_boxes))

        # labels = torch.as_tensor(labels, dtype=torch.int64)

        # Construct the target dictionary
        # target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, boxes
