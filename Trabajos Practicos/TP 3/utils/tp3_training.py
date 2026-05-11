from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .tp3_config import DEVICE, DEVICE_TYPE, EPOCHS, LEARNING_RATE, USE_AMP


def _autocast_context():
    if USE_AMP:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=USE_AMP)
        labels = labels.to(DEVICE, non_blocking=USE_AMP)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context():
            outputs = model(images)
            loss = criterion(outputs, labels)

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / max(1, len(loader)), correct / max(1, total)


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # CAMBIO AQUÍ: Reemplazamos inference_mode() por no_grad() por compatibilidad con DirectML
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=USE_AMP)
            labels = labels.to(DEVICE, non_blocking=USE_AMP)

            with _autocast_context():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / max(1, len(loader)), correct / max(1, total)


def train_model(model, train_loader, val_loader, model_name, epochs=EPOCHS, use_scheduler=False):
    # Optimizaciones de GPU
    if DEVICE_TYPE == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✓ torch.compile() activado")
        except Exception:
            pass  # PyTorch < 2.0 o incompatible
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_path = f"modelos_guardados/{model_name}_best.pt"
    final_path = f"modelos_guardados/{model_name}.pt"
    best_loss = float("inf")
    bad_epochs = 0
    patience = 4
    best_state = None

    print(f"\nEntrenando {model_name}")
    print(f"Dispositivo: {DEVICE}")
    print(f"AMP: {'si' if USE_AMP else 'no'}")

    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
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

        if scheduler is not None:
            scheduler.step(val_loss)

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
