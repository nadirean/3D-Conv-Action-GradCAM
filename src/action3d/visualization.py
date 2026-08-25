"""Plotting and video-overlay visualization helpers."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import KEYPOINT_NAMES, SKELETON_CONNECTIONS


def plot_training_history(history, output_path=None):
    """Plot training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, output_path=None):
    """Plot a confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()
    return cm


def visualize_gradcam_skeleton(input_data, cam, pred_class, true_class, pred_prob, idx_to_label, save_path=None):
    """Visualize a Grad-CAM heatmap for one skeleton sequence.

    Args:
        input_data: Input tensor of shape (3, T, K)
        cam: CAM heatmap of shape (T, K)
        pred_class: Predicted class index
        true_class: True class index
        pred_prob: Prediction probability
        idx_to_label: Mapping from index to original label
        save_path: Path to save figure (optional)
    """
    fig = plt.figure(figsize=(16, 10))

    # 1. Grad-CAM heatmap over time and keypoints
    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(cam.T, aspect="auto", cmap="jet", interpolation="bilinear")
    ax1.set_xlabel("Time Frame")
    ax1.set_ylabel("Keypoint")
    ax1.set_yticks(range(len(KEYPOINT_NAMES)))
    ax1.set_yticklabels(KEYPOINT_NAMES, fontsize=8)
    ax1.set_title(
        f"Grad-CAM Heatmap\nPred: Exercise {idx_to_label[pred_class]} ({pred_prob:.2%}) | "
        f"True: Exercise {idx_to_label[true_class]}"
    )
    plt.colorbar(im, ax=ax1, label="Importance")

    # 2. Temporal importance (mean over keypoints)
    ax2 = fig.add_subplot(2, 2, 2)
    temporal_importance = cam.mean(axis=1)
    ax2.plot(temporal_importance, color="red", linewidth=2)
    ax2.fill_between(range(len(temporal_importance)), temporal_importance, alpha=0.3, color="red")
    ax2.set_xlabel("Time Frame")
    ax2.set_ylabel("Importance")
    ax2.set_title("Temporal Importance (Mean over Keypoints)")
    ax2.grid(True, alpha=0.3)

    # 3. Keypoint importance (mean over time)
    ax3 = fig.add_subplot(2, 2, 3)
    keypoint_importance = cam.mean(axis=0)
    colors = plt.cm.jet(keypoint_importance / keypoint_importance.max())
    ax3.barh(range(len(KEYPOINT_NAMES)), keypoint_importance, color=colors)
    ax3.set_yticks(range(len(KEYPOINT_NAMES)))
    ax3.set_yticklabels(KEYPOINT_NAMES, fontsize=8)
    ax3.set_xlabel("Importance")
    ax3.set_title("Keypoint Importance (Mean over Time)")
    ax3.grid(True, alpha=0.3, axis="x")

    # 4. Skeleton visualization with importance (middle frame)
    ax4 = fig.add_subplot(2, 2, 4)

    input_np = input_data.cpu().numpy()  # (3, T, K)
    mid_frame = input_np.shape[1] // 2
    x_coords = input_np[0, mid_frame, :]
    y_coords = input_np[1, mid_frame, :]
    importance = cam[mid_frame, :]

    for i, j in SKELETON_CONNECTIONS:
        ax4.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], "gray", linewidth=2, alpha=0.5)

    scatter = ax4.scatter(
        x_coords, y_coords, c=importance, cmap="jet", s=200, edgecolors="black", linewidth=1, zorder=5
    )
    plt.colorbar(scatter, ax=ax4, label="Importance")

    for i, name in enumerate(KEYPOINT_NAMES):
        ax4.annotate(name.split("_")[-1][:3], (x_coords[i], y_coords[i]), fontsize=6, ha="center", va="bottom")

    ax4.set_xlim(-0.1, 1.1)
    ax4.set_ylim(1.1, -0.1)  # Invert y-axis for image coordinates
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_title(f"Skeleton (Frame {mid_frame}) with Keypoint Importance")
    ax4.set_aspect("equal")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig


def plot_gradcam_class_summary(gradcam_results, idx_to_label, output_path=None):
    """Plot average temporal/keypoint importance per exercise class.

    Args:
        gradcam_results: List of dicts with 'true_class' and 'cam' (T, K) entries
        idx_to_label: Mapping from index to original label
        save_path/output_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    temporal_arr = np.array([r["cam"].mean(axis=1) for r in gradcam_results])
    im1 = axes[0].imshow(temporal_arr, aspect="auto", cmap="hot", interpolation="bilinear")
    axes[0].set_xlabel("Time Frame")
    axes[0].set_ylabel("Exercise Class")
    axes[0].set_yticks(range(len(gradcam_results)))
    axes[0].set_yticklabels([f"Ex {idx_to_label[r['true_class']]}" for r in gradcam_results])
    axes[0].set_title("Temporal Importance by Exercise Class")
    plt.colorbar(im1, ax=axes[0], label="Importance")

    keypoint_arr = np.array([r["cam"].mean(axis=0) for r in gradcam_results])
    im2 = axes[1].imshow(keypoint_arr, aspect="auto", cmap="hot", interpolation="nearest")
    axes[1].set_xlabel("Keypoint")
    axes[1].set_ylabel("Exercise Class")
    axes[1].set_xticks(range(len(KEYPOINT_NAMES)))
    axes[1].set_xticklabels([kp.replace("_", "\n") for kp in KEYPOINT_NAMES], fontsize=7, rotation=45, ha="right")
    axes[1].set_yticks(range(len(gradcam_results)))
    axes[1].set_yticklabels([f"Ex {idx_to_label[r['true_class']]}" for r in gradcam_results])
    axes[1].set_title("Keypoint Importance by Exercise Class")
    plt.colorbar(im2, ax=axes[1], label="Importance")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def get_video_frames(video_path, start_frame, num_frames):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def create_skeleton_heatmap(cam_frame, keypoint_coords, frame_shape, sigma=30, offset_x=0.0):
    """Create a Gaussian heatmap from per-keypoint importance values.

    Args:
        cam_frame: Importance values for each keypoint
        keypoint_coords: Coordinates of shape (2, K), normalized [0, 1]
        frame_shape: Frame shape (height, width, channels)
        sigma: Gaussian sigma for the heatmap
        offset_x: Horizontal offset in normalized coordinates (0.0-1.0)
    """
    h, w = frame_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    y_grid, x_grid = np.ogrid[:h, :w]

    for kp_idx, importance in enumerate(cam_frame):
        if importance < 0.01:
            continue

        kp_x = int((keypoint_coords[0, kp_idx] + offset_x) * w)
        kp_y = int(keypoint_coords[1, kp_idx] * h)

        kp_x = max(0, min(w - 1, kp_x))
        kp_y = max(0, min(h - 1, kp_y))

        gaussian = np.exp(-((x_grid - kp_x) ** 2 + (y_grid - kp_y) ** 2) / (2 * sigma**2))
        heatmap += importance * gaussian

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def overlay_heatmap_on_frame(frame, heatmap, alpha=0.4):
    """Overlay a heatmap on a video frame."""
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)


def draw_skeleton_on_frame(frame, keypoint_coords, importance=None, offset_x=0.0):
    """Draw the skeleton with keypoints on a frame.

    Args:
        frame: Input frame
        keypoint_coords: Coordinates of shape (2, K), normalized [0, 1]
        importance: Optional importance values for coloring
        offset_x: Horizontal offset in normalized coordinates (0.0-1.0)
    """
    h, w = frame.shape[:2]
    frame_copy = frame.copy()

    for i, j in SKELETON_CONNECTIONS:
        pt1 = (int((keypoint_coords[0, i] + offset_x) * w), int(keypoint_coords[1, i] * h))
        pt2 = (int((keypoint_coords[0, j] + offset_x) * w), int(keypoint_coords[1, j] * h))
        cv2.line(frame_copy, pt1, pt2, (200, 200, 200), 2)

    for kp_idx in range(keypoint_coords.shape[1]):
        pt = (int((keypoint_coords[0, kp_idx] + offset_x) * w), int(keypoint_coords[1, kp_idx] * h))

        if importance is not None:
            imp = importance[kp_idx]
            color = (int(255 * (1 - imp)), 0, int(255 * imp))  # BGR
            radius = int(5 + 10 * imp)
        else:
            color = (0, 255, 0)
            radius = 5

        cv2.circle(frame_copy, pt, radius, color, -1)

    return frame_copy


def visualize_gradcam_on_video(
    video_path,
    skeleton_data,
    cam,
    start_frame,
    pred_class,
    true_class,
    pred_prob,
    idx_to_label,
    num_frames_to_show=8,
    save_path=None,
    offset_x=0.0,
):
    """Visualize Grad-CAM heatmaps overlaid on video frames.

    Args:
        video_path: Path to the source video
        skeleton_data: Skeleton sequence of shape (T, K, 3)
        cam: CAM heatmap of shape (T, K)
        start_frame: First video frame corresponding to the sequence
        pred_class: Predicted class index
        true_class: True class index
        pred_prob: Prediction probability
        idx_to_label: Mapping from index to original label
        num_frames_to_show: Number of frames to display
        save_path: Path to save figure (optional)
        offset_x: Horizontal skeleton offset in normalized coordinates
    """
    sequence_length = cam.shape[0]

    min_len = min(len(skeleton_data), sequence_length)
    skeleton_data = skeleton_data[:min_len]
    cam = cam[:min_len]
    sequence_length = min_len

    frames = get_video_frames(video_path, start_frame, sequence_length)

    if len(frames) == 0:
        print(f"Could not read frames from {video_path}")
        return None

    max_frame_idx = min(len(frames) - 1, len(skeleton_data) - 1, sequence_length - 1)
    frame_indices = np.linspace(0, max_frame_idx, num_frames_to_show, dtype=int)

    fig, axes = plt.subplots(3, num_frames_to_show, figsize=(3 * num_frames_to_show, 9))

    for col, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(frames) or frame_idx >= len(skeleton_data):
            continue

        frame = frames[frame_idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        keypoint_coords = skeleton_data[frame_idx, :, :2].T
        cam_frame = cam[frame_idx, :]

        heatmap = create_skeleton_heatmap(cam_frame, keypoint_coords, frame.shape, offset_x=offset_x)
        frame_overlay_rgb = cv2.cvtColor(overlay_heatmap_on_frame(frame, heatmap, alpha=0.5), cv2.COLOR_BGR2RGB)
        frame_skeleton_rgb = cv2.cvtColor(
            draw_skeleton_on_frame(frame, keypoint_coords, cam_frame, offset_x=offset_x), cv2.COLOR_BGR2RGB
        )

        axes[0, col].imshow(frame_rgb)
        axes[0, col].set_title(f"Frame {start_frame + frame_idx}", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(frame_skeleton_rgb)
        axes[1, col].axis("off")

        axes[2, col].imshow(frame_overlay_rgb)
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=10, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("Skeleton", fontsize=10, rotation=0, ha="right", va="center")
    axes[2, 0].set_ylabel("Grad-CAM", fontsize=10, rotation=0, ha="right", va="center")

    correct_str = "correct" if pred_class == true_class else "incorrect"
    fig.suptitle(
        f"Grad-CAM on Video Frames\n"
        f"Predicted: Exercise {idx_to_label[pred_class]} ({pred_prob:.1%}) | "
        f"True: Exercise {idx_to_label[true_class]} ({correct_str})",
        fontsize=12,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig
