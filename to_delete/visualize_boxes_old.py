import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms


def visualize_random_images(dataset, index):
    """Displays a given number of random images from a dataset along with their bounding boxes,
       accounting for the transforms in the dataset.
    """
    # Get a random sample
    image, annotations = dataset[index]

    # Check if there are transforms in the dataset
    if dataset.transform is not None:
        # Apply the same transforms to the image for visualization
        image = transforms.ToPILImage(mode='RGB')(image)

    # Display the transformed image
    plt.imshow(image)

    # Add bounding boxes to the transformed image
    ax = plt.gca()  # Get current axes
    for ann in annotations:
        if ann != [0, 0, 0, 0]:
            x, y, w, h = ann
            rect = mpatches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.show()


def needs_bbox_adjustment(transform):
    """Helper function to check if a transform potentially modifies bounding boxes"""
    import torchvision.transforms as T
    return isinstance(transform, T.Compose) and any(
        isinstance(t, (T.Resize, T.CenterCrop, T.RandomResizedCrop, T.RandomCrop)) for t in transform.transforms
    )

