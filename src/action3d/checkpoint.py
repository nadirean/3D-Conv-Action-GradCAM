"""Checkpoint saving and loading."""

import torch

from .model import Action3DCNN


def save_checkpoint(
    path, model_state, config, num_classes, label_to_idx, idx_to_label, class_names, best_val_acc, test_acc, history
):
    """Save a complete training checkpoint (model + metadata)."""
    checkpoint = {
        "model_state_dict": model_state,
        "config": config,
        "num_classes": num_classes,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "class_names": class_names,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "history": history,
    }
    torch.save(checkpoint, path)
    print(f"Model saved to: {path}")


def load_checkpoint(path, device):
    """Load a checkpoint and rebuild the model from its stored config.

    Returns (model, checkpoint_dict). The model is in eval mode.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    model = Action3DCNN(
        num_classes=checkpoint["num_classes"],
        sequence_length=config["sequence_length"],
        num_keypoints=config["num_keypoints"],
        hidden_channels=config["hidden_channels"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint
