import torch
import torchvision
from torch import ops


class WeedDetector(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 1. backbone
        self.backbone = torchvision.models.resnet34(pretrained=False)
        # remove the last fully connected layer for classification
        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-1])

        # 2. Region Proposal Network (RPN)
        num_anchors = 9
        self.rpn = torch.nn.Sequential(
            torch.nn.Conv2d(512, 320, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            # objectness classification (2 scores per anchor: object or background)
            torch.nn.Conv2d(320, 2 * num_anchors, kernel_size=1, stride=1),

            # bounding box regression (4 adjustments per anchor)
            torch.nn.Conv2d(320, 4 * num_anchors, kernel_size=1, stride=1),
        )

        # 3. classifier and Regressor Heads
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(320 * 7 * 7, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, num_classes)
        )

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(320 * 7 * 7, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, num_classes * 4)  # *4 for bbox coordinates
        )

    # 4. forward pass method
    def forward(self, images, targets=None):
        x = self.backbone(images)  # Backbone feature extraction
        proposals = self.rpn(x)  # Generate object proposals

        # ROI Align
        pooled_features = ops.roi_align(
            x,  # Input feature map
            proposals,
            output_size=7,  # Output feature map size
            spatial_scale=1.0,  # Feature map scale
        )

        class_logits = self.classifier(pooled_features)
        box_regression = self.regressor(pooled_features)

        # If in training mode, calculate classification and bounding box losses
        if targets is not None:
            loss_classifier, loss_box_reg = self.calculate_losses(class_logits, box_regression, targets)
            return loss_classifier, loss_box_reg

        # If in inference mode, return predictions based on class_logits and box_regression
        return class_logits, box_regression

    # 5. basic training loss function
    def calculate_losses(class_logits, box_regression, targets):
        # Simplified for basic functionality - to be refined
        loss_classifier = torch.nn.CrossEntropyLoss()(class_logits, targets['labels'])
        loss_box_reg = torch.nn.L1Loss()(box_regression, targets['boxes'])
        return loss_classifier, loss_box_reg
