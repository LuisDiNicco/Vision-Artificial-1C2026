import sys

import torch

from .tp3_config import DEVICE


def _print_progress_bar(prefix, current, total):
    # Barra de progreso en una sola linea.
    if total <= 0:
        return
    percent = int((current / total) * 100)
    bar_len = 20
    filled = int(bar_len * current / total)
    bar = "#" * filled + "." * (bar_len - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({percent}%)")
    sys.stdout.flush()


def _clear_progress_bar():
    # Limpia la linea para que no queden barras anteriores.
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


def train_one_epoch(model, loader, criterion, optimizer):
    # Recorre el dataset de entrenamiento una vez (1 epoca).
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)
    update_interval = max(1, total_batches // 20)

    for batch_idx, (images, labels) in enumerate(loader, 1):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_idx % update_interval == 0 or batch_idx == total_batches:
            _print_progress_bar("Train", batch_idx, total_batches)

    _clear_progress_bar()

    avg_loss = running_loss / max(1, len(loader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def evaluate_one_epoch(model, loader, criterion):
    # Evalua sin actualizar pesos, solo para medir performance.
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / max(1, len(loader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy
