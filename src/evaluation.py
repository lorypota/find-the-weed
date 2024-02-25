import torch

from torch.utils.data import DataLoader
from src.train import my_transforms, batch_size, collate_fn, num_workers, device
from src.weed_dataset import DividedWeedDataset

path = "model.pt"
model = torch.load(path)

val_dataset = DividedWeedDataset('_annotations.coco.json', 'dataset/test/', transform=my_transforms)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=num_workers)

# Set the model in evaluation mode
model.eval()


num = 0
element_each_time = 8

# Evaluate the model
with torch.no_grad():
    for list_images, list_targets in val_dataloader:
        num += len(list_images)
        for index in range(len(list_images)):
            for index2 in range(0, len(list_images[index]), element_each_time):
                image = [v.to(device) for v in list_images[index][index2:index2 + 2]]
                target = []
                for dicti in list_targets[index][index2:index2 + 2]:
                    target.append({k: v.to(device) for k, v in dicti.items()})

                # Collect model predictions
                outputs = model(image)

                print(outputs)
                print(target)