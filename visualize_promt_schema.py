"""
Prompt Schema Visualization - Visualize different SAM prompt modes
on a BUSI sample image with separate PNG exports and combined PDF.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from models.wrappers.prompt_utils import resolve_prompt


def _read_uint8_image(path: Path) -> np.ndarray:
    """Read image and ensure RGB format."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]  # Remove alpha channel
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and float(arr.max()) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _read_binary_mask(path: Path) -> np.ndarray:
    """Read mask and convert to binary."""
    arr = _read_uint8_image(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def visualize_prompt_modes(
    image_path: str,
    mask_path: str,
    output_dir: str = "outputs",
    pdf_filename: str = "prompt_schema.pdf",
) -> dict:
    """
    Visualize prompt modes for a single image.
    
    Args:
        image_path: Path to input image
        mask_path: Path to ground-truth mask
        output_dir: Directory to save outputs
        pdf_filename: Name of the combined PDF file
        
    Returns:
        Dictionary with paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image and mask
    image = _read_uint8_image(Path(image_path))
    gt_mask = _read_binary_mask(Path(mask_path))
    
    if image.shape[:2] != gt_mask.shape[:2]:
        raise ValueError(f"Image and mask dimensions don't match: {image.shape[:2]} vs {gt_mask.shape[:2]}")
    
    # Define the three prompt modes
    modes = [
        ("point", "prompt_point"),
        ("bbox", "prompt_bbox"),
        ("point_bbox", "prompt_point_box"),
    ]
    
    output_files = {}
    
    # Create individual PNG files
    for mode_name, prompt_mode in modes:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        # Display image
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(image)
        
        # Resolve and display prompt
        resolved = resolve_prompt(
            {"gt_mask": gt_mask},
            image.shape[:2],
            prompt_mode=prompt_mode,
        )
        
        # Draw bounding box if present
        bbox = resolved.get("bbox")
        if bbox is not None:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            rect = plt.Rectangle(
                (x0, y0),
                max(1, x1 - x0 + 1),
                max(1, y1 - y0 + 1),
                fill=False,
                edgecolor="yellow",
                linewidth=2.5,
            )
            ax.add_patch(rect)
        
        # Draw points if present
        pts = resolved.get("points")
        if pts is not None and np.asarray(pts).size > 0:
            pts_arr = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
            ax.scatter(
                pts_arr[:, 0],
                pts_arr[:, 1],
                c="lime",
                s=60,
                marker="o",
                edgecolors="white",
                linewidths=1.0,
                zorder=5,
            )
        
        ax.axis("off")
        fig.tight_layout(pad=0.1)
        
        # Save PNG
        png_path = output_dir / f"prompt_schema_{mode_name}.png"
        fig.savefig(png_path, dpi=100, bbox_inches='tight', pad_inches=0.05)
        output_files[f"png_{mode_name}"] = str(png_path)
        plt.close(fig)
    
    # Create combined PDF without titles
    pdf_path = output_dir / pdf_filename
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
        
        for ax, (mode_name, prompt_mode) in zip(axes, modes):
            # Display image
            if image.ndim == 2:
                ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(image)
            
            # Resolve and display prompt
            resolved = resolve_prompt(
                {"gt_mask": gt_mask},
                image.shape[:2],
                prompt_mode=prompt_mode,
            )
            
            # Draw bounding box
            bbox = resolved.get("bbox")
            if bbox is not None:
                x0, y0, x1, y1 = [int(v) for v in bbox]
                rect = plt.Rectangle(
                    (x0, y0),
                    max(1, x1 - x0 + 1),
                    max(1, y1 - y0 + 1),
                    fill=False,
                    edgecolor="yellow",
                    linewidth=2.5,
                )
                ax.add_patch(rect)
            
            # Draw points
            pts = resolved.get("points")
            if pts is not None and np.asarray(pts).size > 0:
                pts_arr = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
                ax.scatter(
                    pts_arr[:, 0],
                    pts_arr[:, 1],
                    c="lime",
                    s=60,
                    marker="o",
                    edgecolors="white",
                    linewidths=1.0,
                    zorder=5,
                )
            
            ax.axis("off")
        
        fig.tight_layout(pad=0.5)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    output_files["pdf_combined"] = str(pdf_path)
    
    return output_files


if __name__ == "__main__":
    # BUSI dataset paths
    image_path = r"data\BUSI\Dataset_BUSI_with_GT\benign\benign (10).png"
    mask_path = r"data\BUSI\Dataset_BUSI_with_GT\benign\benign (10)_mask.png"
    
    output_dir = "outputs"
    
    print("Generating prompt schema visualization...")
    
    try:
        files = visualize_prompt_modes(
            image_path=image_path,
            mask_path=mask_path,
            output_dir=output_dir,
            pdf_filename="prompt_schema.pdf",
        )
        
        print("\n✓ Visualization completed successfully!")
        print("\nGenerated files:")
        for key, path in sorted(files.items()):
            print(f"  • {key}: {path}")
        
    except FileNotFoundError as e:
        print(f"✗ Error: File not found: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")
