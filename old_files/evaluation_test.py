from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes
from weed_dataset_visual_and_direction import WeedDatasetEvaluation
from torchmetrics.detection.mean_ap import MeanAveragePrecision


from Constants import *

my_transforms = v2.Compose([
    v2.ToTensor(),
    v2.ConvertBoundingBoxFormat(torchvision.tv_tensors.BoundingBoxFormat.XYXY),
    v2.RandomCrop((128, 128)),
    # v2.RandomRotation((0, 360)),
])

# dataset and dataloader
test_dataset = WeedDatasetEvaluation('_annotations.coco.json', '../dataset/test/', transform=my_transforms)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# loading model
model = torch.load("models/model_v0.pt")
model.eval()

# Instantiate the metric
map_metric = MeanAveragePrecision("xywh", iou_type="bbox")  # Default IoU thresholds

for images, targets in test_dataloader:
    # Load image and convert to suitable format
    image = images[0]
    target = targets[0]

    image_tmp = image.to(torch.uint8)

    # Draw model predictions (with filtering)
    image = image.to(device)
    predictions = model([image])
    predictions = predictions[0]

    """predicted_boxes = []
    for index, score in enumerate(predictions["scores"]):
        if score.item() > 0.7:  # if score is < 0.7, box is likely not accurate
            predicted_boxes.append(predictions["boxes"][index].tolist())

    if len(predicted_boxes) > 0:
        uint8_tensor = (image * 255).clamp(0, 255).to(torch.uint8)
        colors = [(255, 0, 0) for _ in predicted_boxes]
        image = torchvision.old_files.draw_bounding_boxes(uint8_tensor,
                                                      BoundingBoxes(predicted_boxes,
                                                                    canvas_size=(128, 128),
                                                                    format="xyxy").data,
                                                                    colors=colors)
        colors = [(0, 0, 255) for _ in target["boxes"].data.tolist()]
        image = torchvision.old_files.draw_bounding_boxes(image,
                                                      target["boxes"].data,
                                                      colors=colors)
        #visualize
        # v2.functional.to_pil_image(image.to(torch.uint8)).show()
        # input("Enter something to continue")"""

    # Prepare preds for map_metric
    preds = []
    for index, score in enumerate(predictions["scores"]):
        if score.item() > 0.7:
            preds.append({
                'boxes': predictions["boxes"][index],
                'scores': score,
                'labels': predictions["labels"][index]
            })

    # Prepare targets for map_metric
    target = [{
        'boxes': targets[0]['boxes'].data,
        'labels': targets[0]['labels'].data
    }]


    if len(preds) == len(target):
        map_metric.update(preds=preds, target=target)

map_value = map_metric.compute()
print(f"Mean Average Precision (mAP): {map_value}")

# Store results for mAP calculation
# results.append(...)  # Store in the format required by your mAP library

# img, target = train_dataset[867 * 25]
# debug_img(img, target["bboxes"], eval[0]["boxes"], 3)
