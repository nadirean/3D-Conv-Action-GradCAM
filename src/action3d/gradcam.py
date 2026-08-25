"""Grad-CAM implementation for the Action3DCNN model."""

import torch
from scipy.ndimage import zoom


class GradCAM:
    """Grad-CAM for skeleton-based action recognition.

    Generates heatmaps showing which temporal-keypoint regions
    are most important for predictions.
    """

    def __init__(self, model, target_layer_idx=-1):
        """
        Args:
            model: The Action3DCNN model
            target_layer_idx: Index of the conv layer to use for CAM (-1 = last)
        """
        self.model = model
        self.target_layer_idx = target_layer_idx
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        target_layer = self.model.conv_layers[self.target_layer_idx]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Remove registered hooks."""
        for handle in self.hook_handles:
            handle.remove()

    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for input.

        Args:
            input_tensor: Input tensor of shape (1, 3, T, K)
            target_class: Class index to compute CAM for. If None, uses predicted class.

        Returns:
            cam: Heatmap of shape (T, K)
            target_class: Class index the CAM was computed for
            pred_prob: Prediction probability for that class
        """
        self.model.eval()

        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)
        pred_probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        pred_prob = pred_probs[0, target_class].item()

        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations, ReLU, normalize
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)

        cam = cam.squeeze()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.cpu().numpy()
        cam_resized = self._resize_cam(cam, (input_tensor.shape[2], input_tensor.shape[3]))

        return cam_resized, target_class, pred_prob

    @staticmethod
    def _resize_cam(cam, target_size):
        """Resize CAM to target size using bilinear interpolation."""
        h_ratio = target_size[0] / cam.shape[0]
        w_ratio = target_size[1] / cam.shape[1]
        return zoom(cam, (h_ratio, w_ratio), order=1)
