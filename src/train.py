import torch
import torchvision
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import transforms

from model import WeedDetectorCNN
from weed_dataset import DividedWeedDataset


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Concatenates lists of boxes and labels
    targets = [tgt for tgt in targets]

    return images, targets

# """Setup"""
# hyperparameters
num_epochs = 2
learning_rate = 0.001
num_classes = 2  # 1 for weeds, 1 for background
batch_size = 16
num_workers = 4

if torch.cuda.is_available():
    print("CUDA :D")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# transforms
my_transforms = transforms.Compose([
    transforms.ToTensor()
])

# creation of the dataset
train_dataset = DividedWeedDataset('_annotations.coco.json', 'dataset/train/', transform=my_transforms)

# creation of the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=num_workers)

# initialization of model and optimizer
model = WeedDetectorCNN(num_classes).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()  # For classification

def calculate_classification_loss(class_logits, targets):
    return torch.nn.CrossEntropyLoss()(class_logits, targets['labels'])


def calculate_localization_loss(box_regression, targets):
    return torch.nn.L1Loss()(box_regression, targets)


# Load Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                             weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Replace the classifier
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to the right device
model.to(device)

# Training of the model
model.train()
num = 0
element_each_time = 8
length = len(train_dataloader)
for epoch in range(num_epochs):
    for list_images, list_targets in train_dataloader:
        num += len(list_images)
        for index in range(len(list_images)):
            for index2 in range(0, len(list_images[index]), element_each_time):
                image = [v.to(device) for v in list_images[index][index2:index2 + 2]]
                target = []
                for dicti in list_targets[index][index2:index2 + 2]:
                    target.append({k: v.to(device) for k, v in dicti.items()})

                optimizer.zero_grad()
                loss_dict = model(image, target)
                losses: Tensor = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

                del image
                del target

            print(f"Epoch {epoch}, Loss: {losses.item()}")

        print(f"Computed: {num} max: {length}")
        del list_images
        del list_targets

torch.save(model, 'model.pt')
