import math
import os.path

import cv2
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from Constants import *
from weed_dataset_transform import WeedDatasetTransform


def get_direction(generated_boxes):
    result = []
    for box in generated_boxes:
        x_direction = (box[0] + box[2]) / 2
        y_direction = (box[1] + box[3]) / 2

        result.append((round(x_direction), round(y_direction)))

    return result


def show_image_and_boxes(images, outputs, targets):
    for image, output, target in zip(images, outputs, targets):
        image = image.to(torch.uint8)

        boxes = []
        for box, score in zip(output["boxes"], output["scores"]):
            if score.item() >= SCORE_THRESHOLD:
                boxes.append(box.tolist())

        directions = get_direction(boxes)
        if len(boxes) > 0:
            colors = [(0, 0, 255) for _ in boxes]
            image = torchvision.utils.draw_bounding_boxes(image, torch.tensor(boxes), colors=colors, width=2)

        if len(target["boxes"]) > 0:
            colors = [(255, 0, 0) for _ in target["boxes"]]
            image = torchvision.utils.draw_bounding_boxes(image, target["boxes"].data, colors=colors, width=2)

        transform = v2.ToPILImage()
        image = transform(image)

        image_array = np.array(image)

        # calculates angles of direction of the generated boxes
        # this angle is not directly used in the program but can be useful for further implementations
        angles = []
        for direction in directions:
            image_array = cv2.arrowedLine(image_array, (64, 64), direction, (0, 255, 0), 2)
            diff_x = direction[0] - 64
            diff_y = direction[1] - 64

            # calculate angle in radians
            angle_radians = math.atan2(diff_y, diff_x)

            # convert radians to degrees
            angle_degrees = math.degrees(angle_radians)

            angles.append(angle_degrees)

        fig, ax = plt.subplots()
        ax.imshow(image_array)

        plt.title(f"Image Index: test")
        plt.show()


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "models", "model_v6.pt")
    model = torch.load(path)

    root = os.path.join("../dataset", "test")
    val_dataset = WeedDatasetTransform('_annotations.coco.json',
                                       root,
                                       device,
                                       640,
                                       128,
                                       debug=0,
                                       need_weed=True,
                                       rotate=True,
                                       add_mask=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model.eval()
    # evaluates the model
    length = len(val_dataloader) * batch_size
    with torch.no_grad():
        for images_test, targets_test in val_dataloader:

            outputs_test = model(images_test)

            show_image_and_boxes(images_test, outputs_test, targets_test)
