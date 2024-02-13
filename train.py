import torch
import torch.optim as optim
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model import WeedDetector
from weed_dataset import WeedDataset

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
    # This is probably not needed in our case since we just want the image to be cropped in the center
    # transforms.Resize(128),
    transforms.CenterCrop(128),  # Crops from the center to make it 128x128
    transforms.ToTensor()
])

# creation of the datasets
train_dataset = WeedDataset('_annotations.coco.json', 'dataset/train/', transform=my_transforms)
val_dataset = WeedDataset('_annotations.coco.json', 'dataset/test/', transform=my_transforms)

# creation of dataloaders
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# initialization of model and optimizer
model = WeedDetector(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


"""Loss functions"""


def calculate_classification_loss(class_logits, targets):
    return torch.nn.CrossEntropyLoss()(class_logits, targets['labels'])


def calculate_localization_loss(box_regression, targets):
    return torch.nn.L1Loss()(box_regression, targets)


"""Training loop"""
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for images, targets in train_dataloader:
        targets = list(box.to(device) for box in targets)

        optimizer.zero_grad()

        outputs = model(images)

        loss = calculate_localization_loss(outputs, targets)
        loss.backward()

        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")  # Simple loss observation
