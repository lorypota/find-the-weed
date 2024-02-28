import torch

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

num_epochs = 5
learning_rate = 0.00007
num_classes = 2
batch_size = 8
num_workers = 0
SCORE_THRESHOLD = 0.8


def collate_fn(batch):
    batch = [i for i in batch if i != (None, None)]
    images_collate = [item[0] for item in batch]
    targets_collate = [item[1] for item in batch]

    # concatenates lists of boxes and labels
    targets_collate = [tgt for tgt in targets_collate]

    return images_collate, targets_collate
