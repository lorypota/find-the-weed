import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_subimages_and_annotations(dataset, index):
    sub_images, sub_annotations = dataset[index]

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))  # 5x5 grid for sub-images
    axes = axes.flatten()

    for i, (sub_image_tensor, sub_annotations) in enumerate(zip(sub_images, sub_annotations)):
        ax = axes[i]

        print(i, sub_annotations)

        # convert tensor to image
        image_array = sub_image_tensor.numpy()
        if image_array.shape[0] == 3:  # Assuming channels-first
            image_array = image_array.transpose(1, 2, 0)  # Move channels to the end
        ax.imshow(image_array)

        # Draw bounding boxes
        for sub_ann in sub_annotations:
            x, y, w, h = sub_ann
            rect = mpatches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.show()
