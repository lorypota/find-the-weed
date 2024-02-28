import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes

from src.Constants import *
from old_files.weed_dataset_visual_and_direction import WeedDatasetEvaluation

plots_to_show = 5
size = 128
radius = 64


def apply_mask(img):
    mask = torch.zeros(size, size)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    mask_area = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    mask[mask_area] = 1

    mask = mask.unsqueeze(0).repeat(img.size(0), 1, 1)

    mask = mask.to(img.dtype)

    return img * mask


def get_direction(generated_boxes):
    result = []
    for box in generated_boxes:
        x_direction = (box[0] + box[2]) / 2
        y_direction = (box[1] + box[3]) / 2

        result.append((round(x_direction), round(y_direction)))

    return result

my_transforms = v2.Compose([
    v2.ToTensor(),
    v2.ConvertBoundingBoxFormat(torchvision.tv_tensors.BoundingBoxFormat.XYXY),
    v2.RandomCrop((128, 128)),
    v2.RandomRotation((0, 360)),
])

val_dataset = WeedDatasetEvaluation('_annotations.coco.json', '../dataset/test/', transform=my_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

model = torch.load("../src/models/model_v6.pt")
model.eval()

to_img = v2.ToPILImage()
to_tens = v2.PILToTensor()

num_shown_plots = 0

# picks random images and targets in the dataloader
for images, targets in val_dataloader:
    if num_shown_plots >= plots_to_show:
        break

    image = images[0]  # batch is of size 1
    target = targets[0]  # batch is of size 1

    # creates circular masks and puts the image inside of it
    image = apply_mask(image)

    image = image.to(device)
    target = {k: v.to(device) for k, v in target.items()}

    outputs = model([image])

    outputs = outputs[0]

    # collects boxes predicted by the model
    predicted_boxes = []
    for index, score in enumerate(outputs["scores"]):
        if score.item() > 0.7: # if score is < 0.7, box is likely not accurate
            predicted_boxes.append(outputs["boxes"][index].tolist())

    # draw predicted_boxes on top of the masked image
    if len(predicted_boxes) > 0:
        uint8_tensor = (image * 255).clamp(0, 255).to(torch.uint8)
        colors = [(255, 0, 0) for _ in predicted_boxes]
        image = torchvision.utils.draw_bounding_boxes(uint8_tensor,
                                                      BoundingBoxes(predicted_boxes,
                                                                    canvas_size=(128, 128),
                                                                    format="xyxy").data,
                                                                    colors=colors)
        colors = [(0, 0, 255) for _ in target["boxes"].data.tolist()]
        image = torchvision.utils.draw_bounding_boxes(image,
                                                      target["boxes"].data,
                                                      colors=colors)

    # calculate direction of predicted boxes
    res = get_direction(predicted_boxes)

    color = (0, 255, 0)  # Color of the arrow (BGR format)
    thickness = 2  # Thickness of the arrow line
    image = to_img(image)
    np_array = np.array(image)

    directions = []

    for r in res:
        np_array = cv2.arrowedLine(np_array, (64, 64), r, color, thickness)
        diff_x = r[0] - 64
        diff_y = r[1] - 64

        # Calculate angle in radians
        angle_radians = math.atan2(diff_y, diff_x)

        # Convert radians to degrees
        angle_degrees = math.degrees(angle_radians)

        directions.append(angle_degrees)

    if len(res) > 0:
        plt.imshow(np_array)
        num_shown_plots += 1
        plt.axis('off')
        plt.show()
        print(directions)
        print()
