import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .tp3_config import DEVICE, EPOCHS, LEARNING_RATE


def _print_progress(prefix, current, total):
    if total <= 0:
        return
    percent = int((current / total) * 100)
    bar_len = 20
    filled = int(bar_len * current / total)
    bar = "#" * filled + "." * (bar_len - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({percent}%)")
    if current >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def train_one_epoch(model, loader, criterion, optimizer):
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
            _print_progress("Train", batch_idx, total_batches)

    return running_loss / max(1, len(loader)), correct / max(1, total)


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Solo evaluacion: sin gradientes.
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

    return running_loss / max(1, len(loader)), correct / max(1, total)


def train_model(model, train_loader, val_loader, model_name, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    # SGD con momentum para un avance mas estable.
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    models_dir = Path("modelos_guardados")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(models_dir / f"{model_name}_best.pt")
    final_path = str(models_dir / f"{model_name}.pt")
    best_loss = float("inf")
    bad_epochs = 0
    # Criterio de terminacion simple: cortar si no mejora la loss.
    patience = 4
    best_state = None

    print(f"\nEntrenando {model_name}")
    print("Optimizador: SGD")

    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoca {epoch + 1}/{epochs} | "
            f"train {train_acc:.4f}/{train_loss:.4f} | "
            f"val {val_acc:.4f}/{val_loss:.4f} | lr {current_lr:.2e}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping activado")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        torch.save(model.state_dict(), best_path)
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    torch.save(model.state_dict(), final_path)

    elapsed = time.time() - start_time
    print(f"Terminado en {elapsed / 60:.2f} minutos")

    return history


def evaluate_saved_model(model, loader):
    criterion = nn.CrossEntropyLoss()
    return evaluate(model, loader, criterion)


def save_history(history, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def plot_history(history, title, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_acc"], label="train")
    axes[0].plot(epochs, history["val_acc"], label="val")
    axes[0].set_title(f"Accuracy - {title}")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    axes[1].plot(epochs, history["train_loss"], label="train")
    axes[1].plot(epochs, history["val_loss"], label="val")
    axes[1].set_title(f"Loss - {title}")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
