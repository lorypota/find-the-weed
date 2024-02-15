import torch
import torchvision
from torch import ops


class WeedDetectorCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = torchvision.models.resnet34(weights=None)
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-1])

        self.channel_adjustment = torch.nn.Conv2d(512, 18, kernel_size=1, stride=1)

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(100, 4096),  # Input features adjusted
            torch.nn.ReLU(),
            torch.nn.Linear(4096, num_classes)
        )

    def forward(self, images):
        batch_size, num_sub_images, channels, height, width = images.shape

        # Reshape  to make your sub-images compatible with ResNet
        images = images.reshape(batch_size * num_sub_images, channels, height, width)

        x = self.backbone(images)
        x = self.channel_adjustment(x)
        class_logits = self.classifier(x)

        return class_logits

    def calculate_losses(class_logits, box_regression, targets):
        # Simplified for basic functionality - to be refined
        loss_classifier = torch.nn.CrossEntropyLoss()(class_logits, targets['labels'])
        loss_box_reg = torch.nn.L1Loss()(box_regression, targets['boxes'])
        return loss_classifier, loss_box_reg
