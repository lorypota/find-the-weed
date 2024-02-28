import os.path

import torchvision
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights

from Constants import *
from weed_dataset_transform import WeedDatasetTransform

if __name__ == '__main__':
    # creation of the dataset
    root = os.path.join("../dataset", "train")
    train_dataset = WeedDatasetTransform('_annotations.coco.json', root, device,
                                         640, 128, debug=0)

    # creation of the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=num_workers)

    # load Faster R-CNN model
    print("Loading Fast R-CNN Model")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                 weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    grcnn = torchvision.models.detection.faster_rcnn.GeneralizedRCNNTransform(min_size=128, max_size=128,
                                                                              image_mean=[0.485, 0.456, 0.406],
                                                                              image_std=[0.229, 0.224, 0.225])
    model.transform = grcnn

    # replace the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # move model to the right device
    print("Device used: " + str(device))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training of the model
    model.train()
    length = len(train_dataloader) * batch_size
    print("START Training!")
    for epoch in range(num_epochs):
        num = 0
        print(f"Starting Epoch {epoch + 1}, out of {num_epochs}")
        for images, targets in train_dataloader:
            num += len(images)

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            losses: Tensor = sum(loss for loss in loss_dict.values())
            losses.backward()

            print(f"Computed {num} out of {length}, Loss: {losses.item()}")

            optimizer.step()

            del images
            del targets

        print(f"Finished Epoch {epoch + 1}, out of {num_epochs}")

    torch.save(model, 'models/model_v7.pt')
