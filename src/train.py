import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model import WeedDetectorCNN
from weed_dataset import DividedWeedDataset
from visualize_dataset import (visualize_subimages_and_annotations, visualize_images_and_annotations,
                               visualize_subimage_and_annotations)

"""Setup"""
# hyperparameters
num_epochs = 20
batch_size = 4
learning_rate = 0.001
num_classes = 2  # 1 for weeds, 1 for background
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

# creation of the datasets
train_dataset = DividedWeedDataset('_annotations.coco.json', 'dataset/train/', transform=my_transforms)
val_dataset = DividedWeedDataset('_annotations.coco.json', 'dataset/test/', transform=my_transforms)

# visualization of selected images in the train dataset (selected by index)
visualize_subimage_and_annotations(train_dataset, 867 * 25)
visualize_subimages_and_annotations(train_dataset, 867 * 25)
visualize_images_and_annotations('_annotations.coco.json', 'dataset/train/', 867)

# creation of dataloaders
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# initialization of model and optimizer
model = WeedDetectorCNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def calculate_classification_loss(class_logits, targets):
    return torch.nn.CrossEntropyLoss()(class_logits, targets['labels'])


def calculate_localization_loss(box_regression, targets):
    return torch.nn.L1Loss()(box_regression, targets)


for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for sub_images, sub_annotations in train_dataloader:

        optimizer.zero_grad()

        outputs = model(sub_images)

        loss = calculate_localization_loss(outputs, sub_annotations)
        loss.backward()

        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")  # Simple loss observation
