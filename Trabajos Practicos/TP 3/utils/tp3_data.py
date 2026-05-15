import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from .tp3_config import BATCH_SIZE, IMG_SIZE, TEST_PATH, TRAIN_PATH, USE_PIN_MEMORY, get_num_workers


def build_transforms(use_augmentation: bool):
    # Normalizacion estandar para imagenes RGB.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Transformacion base usada para validacion y test.
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    if not use_augmentation:
        return eval_transform, eval_transform

    # Aumentacion basica para que el modelo vea mas variedad de ejemplos.
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


def build_loaders(use_augmentation: bool = False, batch_size: int = BATCH_SIZE):
    # Prepara transforms y carga los datasets desde disco.
    train_transform, eval_transform = build_transforms(use_augmentation)
    num_workers = get_num_workers()

    full_train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_PATH, transform=eval_transform)

    # Dividimos entrenamiento en train/val para medir generalizacion.
    train_size = int(len(full_train_dataset) * 0.95)
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": USE_PIN_MEMORY,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
