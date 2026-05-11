from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .tp3_config import BATCH_SIZE, IMG_SIZE, TEST_PATH, TRAIN_PATH, USE_AMP, get_num_workers


def _build_transforms(use_augmentation: bool):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    if not use_augmentation:
        return test_transform, test_transform

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, test_transform


def build_loaders(use_augmentation: bool = False, batch_size: int = BATCH_SIZE):
    train_transform, test_transform = _build_transforms(use_augmentation)
    num_workers = get_num_workers()

    train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_PATH, transform=test_transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": USE_AMP,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, test_loader
