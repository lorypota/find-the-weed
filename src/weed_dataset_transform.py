import json
import os
import random

import PIL
import numpy as np
import torch
import torchvision.utils
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat


def debug_img(img, target, debug):
    image_tmp = img.to(torch.uint8)
    if len(target["boxes"]) > 0:
        colors = [(255, 0, 0) for _ in target["boxes"]]
        image_tmp = torchvision.utils.draw_bounding_boxes(image_tmp, target["boxes"].data, colors=colors, width=2)
    v2.functional.to_pil_image(image_tmp.to(torch.uint8)).show()
    if debug == 3:
        input("Enter something to continue")


class WeedDatasetTransform(Dataset):
    # debug=0 doesn't do any debug
    # debug=1 shows full + result image at index without waiting for keyboard input
    # debug=2 shows full + result image at index and all of the steps in between without waiting for keyboard input
    # debug=3 shows full + result image at index and waits for keyboard input after each image
    def __init__(self, ann_file, root, device, start_size, resized_size, always_centered=False, need_weed=True,
                 debug=0, debug_data=False, rotate=True, add_mask=True):
        self.root = root
        self.device = device
        self.start_size = start_size
        self.resized_size = resized_size
        self.always_centered = always_centered
        self.need_weed = need_weed
        self.debug = debug
        self.debug_data = debug_data
        self.rotate = rotate
        self.add_mask = add_mask
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

        if len(boxes) == 0:
            print(f"Found image with no weeds with index: {index}")
            return None, None

        # each bbox has a label of 1
        labels = torch.ones((len(boxes),), dtype=torch.int64, device=self.device)

        # store bboxes with specific type
        boxes = BoundingBoxes(boxes, format=BoundingBoxFormat.XYWH, canvas_size=(self.start_size, self.start_size),
                              dtype=torch.float32, device=self.device)

        # converts from (x, y, w, h) to (xstart, ystart, xend, yend)
        boxes = v2.ConvertBoundingBoxFormat(BoundingBoxFormat.XYXY)(boxes)

        target["boxes"] = boxes
        target["labels"] = labels

        # removes bbox that degenerate (bbox outside of cropped image)
        sanitize_bbox = v2.SanitizeBoundingBoxes()

        # transforms image to tensor of type float (required by the model)
        image = v2.functional.pil_to_tensor(image).to(torch.float32).to(self.device)

        if self.debug >= 1:
            debug_img(image, target, self.debug)

        loops = 0
        if self.always_centered:
            loops = 100
        while True:
            loops += 1
            if loops > 101:
                print(f"Found image with no weeds bugged with index: {index}")
                return None, None
            # if after 100 attempts none of the selected cropped images had annotations,
            # it tries to center the cropping around an annotated weed
            if loops > 100:
                tmp = target["boxes"]
                box = tmp[random.randint(0, len(tmp) - 1)]
                x1, y1, x2, y2 = box
                x0 = max(x2 - x1 - (self.resized_size / 2), 0)
                y0 = max(y2 - y1 - (self.resized_size / 2), 0)
                if isinstance(x0, Tensor):
                    x0 = round(x0.item())
                if isinstance(y0, Tensor):
                    y0 = round(y0.item())
                result_img = v2.functional.crop_image(image, y0, x0, self.resized_size, self.resized_size)
                result_target = target
                bbox, canvas_size = v2.functional.crop_bounding_boxes(target["boxes"], BoundingBoxFormat.XYXY, y0, x0,
                                                                      self.resized_size, self.resized_size)
                result_target["boxes"] = BoundingBoxes(bbox, format=BoundingBoxFormat.XYXY, canvas_size=canvas_size,
                                                       device=self.device, dtype=torch.float32)
            else:
                # random crop of the image
                result_img, result_target = v2.RandomCrop((self.resized_size, self.resized_size))(image, target)

            result_target = sanitize_bbox(result_target)

            if self.debug >= 2:
                debug_img(result_img, result_target, self.debug)

            if self.add_mask:
                mask = torch.zeros(self.resized_size, self.resized_size)
                center = self.resized_size // 2
                y, x = np.ogrid[:self.resized_size, :self.resized_size]
                mask_area = (x - center) ** 2 + (y - center) ** 2 <= (self.resized_size / 2) ** 2
                mask[mask_area] = 1

                # Repeat the mask for each channel if the input tensor is multichannel
                mask = mask.unsqueeze(0).repeat(result_img.size(0), 1, 1)

                # Convert mask to the same dtype as the input tensor
                mask = mask.to(result_img.dtype).to(self.device)

                # Apply the mask to the input tensor
                result_img = result_img * mask

                if self.debug >= 2:
                    debug_img(result_img, result_target, self.debug)

            if self.rotate:
                # randomly rotates the image
                result_img, result_target = v2.RandomRotation((0, 360))(result_img, result_target)

                if self.debug >= 2:
                    debug_img(result_img, result_target, self.debug)

            result_target = sanitize_bbox(result_target)

            if not self.need_weed or len(result_target["boxes"]) > 0:
                break

        if self.debug >= 1:
            debug_img(result_img, result_target, self.debug)

        if self.debug_data:
            result_target["imageId"] = torch.tensor(index, dtype=torch.int64, device=self.device)

        del image
        del target

        # print(index)

        return result_img, result_target
