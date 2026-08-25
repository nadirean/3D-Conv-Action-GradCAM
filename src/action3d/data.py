"""Data loading, sequence creation and PyTorch dataset utilities."""

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import DEFAULT_CONFIG, FLIP_SWAP_PAIRS


def load_skeleton_file(filepath):
    """Load skeleton CSV file and convert to numpy array.

    Returns array of shape (num_frames, num_keypoints, 3) and the landmark order.
    """
    df = pd.read_csv(filepath, low_memory=False)

    if "frame_number" not in df.columns:
        df.columns = ["frame_number", "landmark", "x", "y", "z"]

    landmarks_order = df[df["frame_number"] == df["frame_number"].min()]["landmark"].tolist()
    num_keypoints = len(landmarks_order)
    frames = sorted(df["frame_number"].unique())
    num_frames = len(frames)

    skeleton_array = np.zeros((num_frames, num_keypoints, 3), dtype=np.float32)

    for frame_idx, frame_num in enumerate(frames):
        frame_data = df[df["frame_number"] == frame_num]
        for kp_idx, landmark in enumerate(landmarks_order):
            kp_data = frame_data[frame_data["landmark"] == landmark]
            if len(kp_data) > 0:
                skeleton_array[frame_idx, kp_idx, 0] = kp_data["x"].values[0]
                skeleton_array[frame_idx, kp_idx, 1] = kp_data["y"].values[0]
                skeleton_array[frame_idx, kp_idx, 2] = kp_data["z"].values[0]

    return skeleton_array, landmarks_order


def load_label_file(filepath):
    """Load label CSV file. Returns array of labels for each frame (-1 = no activity)."""
    df = pd.read_csv(filepath, header=None, low_memory=False)
    labels = pd.to_numeric(df.iloc[:, -1], errors="coerce").fillna(-1).astype(int).values
    return labels


def get_matching_files(skeleton_dir, label_dir):
    """Get list of file stems that exist in both skeleton and label directories."""
    skeleton_files = {f.replace(".csv", "") for f in os.listdir(skeleton_dir) if f.endswith(".csv")}
    label_files = {f.replace(".csv", "") for f in os.listdir(label_dir) if f.endswith(".csv")}
    return sorted(skeleton_files.intersection(label_files))


def load_split_file(split_file):
    """Load the dataset split definition. Returns (train_files, test_files) sets."""
    split_df = pd.read_csv(split_file, header=None, names=["file_name", "split"])
    train_files = set(split_df[split_df["split"] == "train"]["file_name"].tolist())
    test_files = set(split_df[split_df["split"] == "test"]["file_name"].tolist())
    return train_files, test_files


def create_sequences(skeleton_data, labels, sequence_length, stride, exclude_no_activity=True):
    """Create sequences from skeleton data using a sliding window.

    Args:
        skeleton_data: Array of shape (num_frames, num_keypoints, 3)
        labels: Array of labels for each frame
        sequence_length: Number of frames per sequence
        stride: Step size for the sliding window
        exclude_no_activity: If True, skip sequences with only no-activity labels

    Returns:
        sequences: List of arrays, each (sequence_length, num_keypoints, 3)
        sequence_labels: List of labels (majority vote for each sequence)
    """
    sequences = []
    sequence_labels = []
    num_frames = len(skeleton_data)

    for start_idx in range(0, num_frames - sequence_length + 1, stride):
        end_idx = start_idx + sequence_length
        seq_data = skeleton_data[start_idx:end_idx]
        seq_labels = labels[start_idx:end_idx]

        exercise_labels = seq_labels[seq_labels != -1]
        if len(exercise_labels) > 0:
            label_counts = Counter(exercise_labels)
            majority_label = label_counts.most_common(1)[0][0]
        else:
            if exclude_no_activity:
                continue
            majority_label = -1

        sequences.append(seq_data)
        sequence_labels.append(majority_label)

    return sequences, sequence_labels


def load_all_sequences(skeleton_dir, label_dir, split_file, config=None):
    """Load every matching recording and build train/test sequences.

    The train/test assignment follows split.csv; a stratified train/validation
    split is carved out of the training files.

    Returns dict with X_train, X_val, X_test, y_train, y_val, y_test,
    label_to_idx, idx_to_label and num_classes.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    skeleton_dir, label_dir, split_file = Path(skeleton_dir), Path(label_dir), Path(split_file)

    train_files, test_files = load_split_file(split_file)
    matching_files = [f for f in get_matching_files(skeleton_dir, label_dir) if f in train_files | test_files]

    train_sequences, train_labels = [], []
    test_sequences, test_labels = [], []

    for file_name in matching_files:
        skeleton_data, _ = load_skeleton_file(skeleton_dir / f"{file_name}.csv")
        labels = load_label_file(label_dir / f"{file_name}.csv")

        min_len = min(len(skeleton_data), len(labels))
        skeleton_data = skeleton_data[:min_len]
        labels = labels[:min_len]

        sequences, seq_labels = create_sequences(
            skeleton_data, labels, cfg["sequence_length"], cfg["stride"], exclude_no_activity=True
        )

        if file_name in train_files:
            train_sequences.extend(sequences)
            train_labels.extend(seq_labels)
        else:
            test_sequences.extend(sequences)
            test_labels.extend(seq_labels)

    X_train_all = np.array(train_sequences, dtype=np.float32)
    y_train_all = np.array(train_labels, dtype=np.int64)
    X_test = np.array(test_sequences, dtype=np.float32)
    y_test_orig = np.array(test_labels, dtype=np.int64)

    unique_labels = sorted(set(np.concatenate([y_train_all, y_test_orig]).tolist()))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    y_train_all_mapped = np.array([label_to_idx[label] for label in y_train_all])
    y_test = np.array([label_to_idx[label] for label in y_test_orig])

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all_mapped, test_size=cfg["val_size"], random_state=42, stratify=y_train_all_mapped
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "num_classes": len(unique_labels),
    }


class SkeletonDataset(Dataset):
    """PyTorch Dataset for skeleton sequences.

    Args:
        sequences: numpy array of shape (N, T, K, 3) - T=temporal, K=keypoints
        labels: numpy array of shape (N,)
        augment: whether to apply data augmentation
        channel_mode: "xyz" uses raw coordinates as channels (default,
            compatible with the released checkpoint); "velocity" uses
            frame-to-frame finite differences instead.
    """

    def __init__(self, sequences, labels, augment=False, channel_mode="xyz"):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.channel_mode = channel_mode

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx].copy()  # (T, K, 3)
        y = self.labels[idx]

        if self.augment:
            x = self._augment(x)

        if self.channel_mode == "velocity":
            x = self._to_velocity(x)

        # Transpose to (C, T, K): coordinates as channels, (T, K) as spatial dims
        x = x.transpose(2, 0, 1)  # (3, T, K)

        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()

    @staticmethod
    def _to_velocity(x):
        """Convert coordinates to frame-to-frame velocity (first frame zeroed)."""
        velocity = np.zeros_like(x, dtype=np.float32)
        velocity[1:] = np.diff(x, axis=0)
        return velocity

    def _augment(self, x):
        """Apply data augmentation to a skeleton sequence."""
        # Random horizontal flip (mirror skeleton, swap left/right keypoints)
        if np.random.random() > 0.5:
            x[:, :, 0] = 1 - x[:, :, 0]
            for i, j in FLIP_SWAP_PAIRS:
                x[:, i, :], x[:, j, :] = x[:, j, :].copy(), x[:, i, :].copy()

        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.01, x.shape).astype(np.float32)
            x = x + noise

        # Random scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

        return x


def build_dataloaders(data, config=None, channel_mode="xyz"):
    """Build train/val/test DataLoaders from a load_all_sequences result."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    train_dataset = SkeletonDataset(data["X_train"], data["y_train"], augment=True, channel_mode=channel_mode)
    val_dataset = SkeletonDataset(data["X_val"], data["y_val"], augment=False, channel_mode=channel_mode)
    test_dataset = SkeletonDataset(data["X_test"], data["y_test"], augment=False, channel_mode=channel_mode)

    return {
        "train": DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0),
        "val": DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0),
        "test": DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0),
        "test_dataset": test_dataset,
    }
