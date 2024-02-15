import os
import json
import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def convert_and_draw(ax, sub_image, target):
    # convert tensor to image
    image_array = sub_image.numpy()
    if image_array.shape[0] == 3:  # Assuming channels-first
        image_array = image_array.transpose(1, 2, 0)  # Move channels to the end
    ax.imshow(image_array)

    # Draw bounding boxes
    if np.array(target["labels"]) != [0]:
        for sub_ann in np.array(target["boxes"]):
            x, y, w, h = sub_ann
            rect = mpatches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)


def visualize_subimage_and_annotations(dataset, sub_img_index):
    sub_image, target = dataset[sub_img_index]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    convert_and_draw(ax, sub_image, target)

    plt.show()


def visualize_subimages_and_annotations(dataset, index):
    img_index = index // 25

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))  # 5x5 grid
    axes = axes.flatten()

    for i in range(25):
        sub_image, target = dataset[img_index * 25 + i]
        ax = axes[i]

        convert_and_draw(ax, sub_image, target)

    plt.show()


def visualize_images_and_annotations(ann_file, root, index):
    with open(os.path.join(root, ann_file)) as file:
        ann_json = json.load(file)

    img_path = os.path.join(root, ann_json["images"][index]["file_name"])
    image = PIL.Image.open(img_path)

    anns_for_img = []
    for jsonObj in ann_json["annotations"]:
        if jsonObj["image_id"] == index:
            anns_for_img.append(jsonObj)

    image_array = np.array(image)

    fig, ax = plt.subplots()
    ax.imshow(image_array)

    for ann in anns_for_img:
        x, y, w, h = ann['bbox']
        rect = mpatches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.title(f"Image Index: {index}")
    plt.show()
