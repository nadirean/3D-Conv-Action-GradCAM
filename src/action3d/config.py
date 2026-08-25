"""Shared configuration and constants for the Action3D pipeline."""

DEFAULT_CONFIG = {
    # Data parameters
    "sequence_length": 64,
    "stride": 32,
    "num_keypoints": 17,
    "num_coords": 3,
    # Training parameters
    "batch_size": 32,
    "learning_rate": 1e-3,
    "num_epochs": 50,
    "patience": 10,
    # Model parameters
    "hidden_channels": [32, 64, 128],
    "dropout": 0.3,
    # Data split
    "test_size": 0.2,
    "val_size": 0.1,
}

# YOLO pose keypoint order
KEYPOINT_NAMES = [
    "Nose",
    "L_Eye",
    "R_Eye",
    "L_Ear",
    "R_Ear",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
]

# Keypoint index pairs forming the skeleton graph (YOLO pose format)
SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Head
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # Arms
    (5, 11),
    (6, 12),
    (11, 12),  # Torso
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # Legs
]

# Index pairs swapped by the horizontal-flip augmentation
FLIP_SWAP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
