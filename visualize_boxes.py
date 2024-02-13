import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_random_images(dataset, num_images=3):
    """Displays a given number of random images from a dataset along with their bounding boxes"""

    for _ in range(num_images):
        # Get a random sample
        sample = random.choice(dataset)
        image = sample['image']
        annotations = sample['annotations']

        # Display the image
        plt.imshow(image)

        # Add bounding boxes
        ax = plt.gca()  # Get current axes
        for annot in annotations:
            x, y, w, h = annot['bbox']
            rect = mpatches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.show()
