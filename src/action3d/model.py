"""Action3DCNN model architecture."""

import torch.nn as nn


class Action3DCNN(nn.Module):
    """3D-CNN for skeleton-based action recognition.

    Input shape: (batch, 3, T, K) where:
        - 3: coordinate channels (x, y, z)
        - T: temporal dimension (sequence_length)
        - K: keypoints (17)

    We treat this as a 2D problem with 3 channels (like an RGB image)
    where the spatial dimensions are (T, K) - temporal x keypoints.
    This avoids issues with 3D pooling making dimensions too small.
    """

    def __init__(self, num_classes, sequence_length=64, num_keypoints=17, hidden_channels=None, dropout=0.3):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]

        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints

        self.conv_layers = nn.ModuleList()

        in_ch = 3  # coordinate channels
        for out_ch in hidden_channels:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            in_ch = out_ch

        self._calc_flat_size(sequence_length, num_keypoints)

        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # For Grad-CAM - store intermediate features
        self.gradients = None
        self.activations = None

    def _calc_flat_size(self, T, K):
        """Calculate flattened size after all conv layers."""
        h, w = T, K
        for _ in self.conv_layers:
            h = h // 2
            w = max(w // 2, 1)
        self.flat_size = self.conv_layers[-1][0].out_channels * h * w

    def activations_hook(self, grad):
        """Hook for gradients."""
        self.gradients = grad

    def forward(self, x):
        for i, conv_block in enumerate(self.conv_layers):
            x = conv_block(x)

            # Store activations from the last conv layer for Grad-CAM
            if i == len(self.conv_layers) - 1:
                self.activations = x
                if x.requires_grad:
                    x.register_hook(self.activations_hook)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations
