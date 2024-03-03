import os

import torch
from torch.utils.data import DataLoader

from src.Constants import *
from src.validate import show_image_and_boxes
from src.weed_dataset_transform import WeedDatasetTransform


def calculate_iou(pred, correct):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        pred (tensor): Bounding box 1, format: (x1, y1, x2, y2)
        correct (tensor): Bounding box 2, format: (x1, y1, x2, y2)

    Returns:
        float: IoU value
    """
    # if prediction is contained completely in truth, box is likely accurate due to cropping
    if pred[0] > correct[0] and pred[1] > correct[1] and pred[2] < correct[2] and pred[3] < correct[3]:
        return 1

    # determine the overlap region
    x_overlap = max(0, min(pred[2], correct[2]) - max(pred[0], correct[0]))
    y_overlap = max(0, min(pred[3], correct[3]) - max(pred[1], correct[1]))
    intersection_area = x_overlap * y_overlap

    # calculate union area
    union_area = (pred[2] - pred[0]) * (pred[3] - pred[1]) + \
                 (correct[2] - correct[0]) * (correct[3] - correct[1]) - intersection_area

    return intersection_area / union_area


def reasonable_sized_area(bbox):
    # there are cases in which the area of the weed might be too small due to the applied circular mask

    # if one side of the rectangle is too small
    if bbox[2] - bbox[0] < 40 or bbox[3] - bbox[1] < 40:
        return False

    # if one of the angles of the bbox is near to the angles of the image and the area is small (too much black space)
    if (bbox[0] < 15 and bbox[1] < 15) or (bbox[0] < 15 and bbox[3] > 113) or (bbox[2] > 113 and bbox[3] > 113) or (
            bbox[2] > 113 and bbox[1] < 15):
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 2000:
            return False

    return True


def evaluate_model(dataloader, visualize=False, iou_threshold=0.5):
    true_positives = [0, 0, 0, 0, 0]
    false_positives = [0, 0, 0, 0, 0]
    false_negatives = [0, 0, 0, 0, 0]

    score_threshold = 0.7

    with torch.no_grad():
        for images, targets in dataloader:
            for i in range(5):
                model_name = "model_v" + str(i) + ".pt"
                path = os.path.join(os.getcwd(), "models", model_name)
                model = torch.load(path)
                model.eval()

                outputs = model(images)

                if visualize:
                    show_image_and_boxes(images, outputs, targets)

                for image, output, target in zip(images, outputs, targets):
                    predicted_boxes = []
                    for box, score in zip(output["boxes"], output["scores"]):
                        if score.item() >= score_threshold:
                            predicted_boxes.append(box.tolist())

                    # extracts each correct annotation
                    for correct_box in target["boxes"]:
                        if not reasonable_sized_area(correct_box):
                            pass

                        best_iou = 0
                        # extracts each prediction
                        for pred_bbox in predicted_boxes:
                            iou = calculate_iou(pred_bbox, correct_box)
                            iou = iou.item() if type(iou) == torch.Tensor else iou
                            if iou > best_iou:
                                best_iou = iou

                        # print(best_iou)
                        if best_iou > iou_threshold:
                            true_positives[i] += 1
                        else:
                            false_negatives[i] += 1

                    false_positives[i] += len(predicted_boxes)

    ret_eval = []

    for i in range(5):
        false_positives[i] = false_positives[i] - true_positives[i]

        if (true_positives[i] + false_positives[i]) > 0:
            precision = true_positives[i] / (true_positives[i] + false_positives[i])
        else:
            precision = 0

        if (true_positives[i] + false_negatives[i]) > 0:
            recall = true_positives[i] / (true_positives[i] + false_negatives[i])
        else:
            recall = 0

        ret_eval.append((precision, recall, true_positives[i], false_positives[i], false_negatives[i]))

    return ret_eval


if __name__ == "__main__":

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

    model_eval = evaluate_model(val_dataloader, False)

    print('|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|'.format("model", "precision", "recall", "TP", "FP", "FN"))
    print('-' * 66)

    for i in range(5):
        # Round the first two elements to 3 decimal places
        rounded_values = [round(x, 3) if isinstance(x, float) else x for x in model_eval[i]]
        print('|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|'.format("n." + str(i), *rounded_values))
        print('-' * 66)
