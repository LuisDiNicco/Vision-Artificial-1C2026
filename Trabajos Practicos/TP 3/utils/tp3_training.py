import time
from pathlib import Path

import torch
import torch.nn as nn

from .tp3_config import DEVICE, EPOCHS, LEARNING_RATE
from .tp3_training_core import evaluate_one_epoch, train_one_epoch


def _create_optimizer(model):
    # SGD con momentum para un avance mas estable.
    return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)


def _copy_state_dict(model):
    # Copia segura a CPU para guardar el mejor modelo.
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_model(model, train_loader, val_loader, model_name, epochs=EPOCHS):
    # Pipeline basico: entrenar, validar, guardar el mejor modelo.
    criterion = nn.CrossEntropyLoss()
    optimizer = _create_optimizer(model)

    # Guardamos el historial para poder graficar y explicar el progreso.
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    models_dir = Path("modelos_guardados")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(models_dir / f"{model_name}_best.pt")
    final_path = str(models_dir / f"{model_name}.pt")
    best_loss = float("inf")
    bad_epochs = 0
    patience = 4
    best_state = None

    print(f"\nEntrenando {model_name}")
    print("Optimizador: SGD")

    start_time = time.time()

    # Cada epoca = entrenamiento + validacion completa.
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion)

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

        # Si la validacion mejora, guardamos el modelo.
        if val_loss < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            best_state = _copy_state_dict(model)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                # Si no mejora varias epocas, frenamos para evitar sobreajuste.
                print("Early stopping activado")
                break

    # Restauramos el mejor estado y guardamos el modelo final.
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
    # Evalua el modelo final en el conjunto de prueba.
    criterion = nn.CrossEntropyLoss()
    return evaluate_one_epoch(model, loader, criterion)
