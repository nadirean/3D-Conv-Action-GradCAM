"""Training and evaluation loops."""

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .model import Action3DCNN


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch. Returns (epoch_loss, epoch_acc)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model. Returns (epoch_loss, epoch_acc, all_preds, all_labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


def make_criterion_optimizer(model, y_train, config, device):
    """Build the class-weighted loss, optimizer and scheduler."""
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    return criterion, optimizer, scheduler


def train_model(model, loaders, y_train, config, device):
    """Run the full training loop with early stopping.

    Returns (best_model_state, history, best_val_acc).
    """
    criterion, optimizer, scheduler = make_criterion_optimizer(model, y_train, config, device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print("Starting training...")
    print("=" * 70)

    for epoch in range(config["num_epochs"]):
        train_loss, train_acc = train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{config['num_epochs']}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  -> New best model. Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break

    print("=" * 70)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")

    return best_model_state, history, best_val_acc


def build_model(config, num_classes, device):
    """Create an Action3DCNN from a config dict and move it to device."""
    return Action3DCNN(
        num_classes=num_classes,
        sequence_length=config["sequence_length"],
        num_keypoints=config["num_keypoints"],
        hidden_channels=config["hidden_channels"],
        dropout=config["dropout"],
    ).to(device)
