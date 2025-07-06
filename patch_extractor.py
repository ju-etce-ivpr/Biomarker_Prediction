import os
import openslide
import numpy as np
from PIL import Image


def is_informative(patch, threshold=0.8):
    """
    Check if patch has enough tissue. Discard white/empty patches.
    """
    gray = np.array(patch.convert("L")) / 255.0
    return (gray < 0.95).mean() > threshold


def extract_patches(slide_path, output_dir, patch_size=224, level=0, max_patches=None):
    """
    Extract informative patches from a WSI file at a given level (magnification).
    """
    slide = openslide.OpenSlide(slide_path)
    width, height = slide.level_dimensions[level]
    basename = os.path.splitext(os.path.basename(slide_path))[0]
    
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = slide.read_region((x * slide.level_downsamples[level], y * slide.level_downsamples[level]), level, (patch_size, patch_size)).convert("RGB")
            if is_informative(patch):
                patch_name = f"{basename}_L{level}_x{x}_y{y}.png"
                patch.save(os.path.join(output_dir, patch_name))
                count += 1
            if max_patches and count >= max_patches:
                return


def process_wsi_directory(wsi_dir, patch_output_root, patch_size=224, levels=(0, 2), max_patches_per_level=(2000, 200)):
    """
    Processes all WSI files in a directory and saves extracted patches.
    levels: tuple of levels (e.g., (0, 2)) corresponding to 20x and 5x magnification.
    """
    for wsi_file in os.listdir(wsi_dir):
        if not wsi_file.endswith(".svs"):
            continue

        wsi_path = os.path.join(wsi_dir, wsi_file)
        for i, level in enumerate(levels):
            output_dir = os.path.join(patch_output_root, f"level_{level}")
            extract_patches(
                slide_path=wsi_path,
                output_dir=output_dir,
                patch_size=patch_size,
                level=level,
                max_patches=max_patches_per_level[i]
            )
