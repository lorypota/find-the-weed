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
        pil_image = Image.fromarray(image)

        annotations = []
        for jsonObj in self.json["annotations"]:
            if jsonObj["image_id"] == idx:
                annotations.append(jsonObj)

        boxes = []
        # labels = []
        for ann in annotations:
            xmin, ymin, w, h = ann['bbox']
            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])  # Create bounding box tensor

        # Transform data into proper tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Padding
        num_boxes = len(boxes)
        max_detections = 10 # TODO: change handling of max_detections
        if num_boxes < max_detections:
            boxes = torch.nn.functional.pad(boxes, (0, 0, 0, max_detections - num_boxes))

        if self.transform:
            image = self.transform(pil_image)

            image_width, image_height = (128, 128)

            print(boxes)

            for box in boxes:
                x1, y1, x2, y2 = box.tolist()

                # Calculate how much was cropped:
                crop_left = (image_width - 128) // 2
                crop_top = (image_height - 128) // 2

                # Adjust bounding box coordinates
                box[0] = max(0, x1 - crop_left)
                box[1] = max(0, y1 - crop_top)
                box[2] = min(image_width, x2 - crop_left)
                box[3] = min(image_height, y2 - crop_top)

        return image, boxes
