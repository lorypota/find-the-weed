import torch
from torchvision.transforms import transforms

from weed_dataset import DividedWeedDataset
from visualize_boxes import visualize_subimages_and_annotations

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

# creation of transforms
my_transforms = transforms.Compose([
    transforms.CenterCrop(128),  # Crops from the center to make it 128x128
    transforms.ToTensor()
])

# creation of the datasets
train_dataset = DividedWeedDataset('_annotations.coco.json', 'dataset/train/', transform=my_transforms)
val_dataset = DividedWeedDataset('_annotations.coco.json', 'dataset/test/', transform=my_transforms)

# visualization of selected images in the train dataset (selected by index)
visualize_subimages_and_annotations(train_dataset, 867)
# TODO: check if annotations are accurate by comparing full image with annotations to sub-images with sub-annotations

"""# creation of dataloaders
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# initialization of model and optimizer
model = WeedDetector(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def calculate_classification_loss(class_logits, targets):
    return torch.nn.CrossEntropyLoss()(class_logits, targets['labels'])


def calculate_localization_loss(box_regression, targets):
    return torch.nn.L1Loss()(box_regression, targets)


for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for images, targets in train_dataloader:
        targets = list(box.to(device) for box in targets)

        optimizer.zero_grad()

        outputs = model(images)

        loss = calculate_localization_loss(outputs, targets)
        loss.backward()

        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")  # Simple loss observation"""
