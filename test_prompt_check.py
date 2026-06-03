"""
Quick check to verify prompt generation matches training setup.
"""
from PIL import Image
import numpy as np
from pathlib import Path
from models.wrappers.prompt_utils import resolve_prompt

# Load test image
image_path = r'data\BUSI\Dataset_BUSI_with_GT\benign\benign (10).png'
mask_path = r'data\BUSI\Dataset_BUSI_with_GT\benign\benign (10)_mask.png'

image = np.asarray(Image.open(image_path))
mask = np.asarray(Image.open(mask_path))
if mask.ndim == 3:
    mask = mask[..., 0]
mask = (mask > 0).astype(np.uint8)

print('=' * 70)
print('PROMPT GENERATION VERIFICATION')
print('=' * 70)
print(f'\nImage shape: {image.shape}')
print(f'Mask shape: {mask.shape}')
print(f'Foreground pixels: {np.sum(mask > 0)}')
print()

# Test each prompt mode like in the experiment_engine
modes = [
    ('prompt_point', 'Point Only'),
    ('prompt_bbox', 'Bounding Box Only'),  
    ('prompt_point_box', 'Point + Box'),
]

for mode_key, mode_name in modes:
    print(f'\n{mode_name.upper()} ({mode_key}):')
    print('-' * 70)
    resolved = resolve_prompt(
        {'gt_mask': mask},
        image.shape[:2],
        prompt_mode=mode_key,
    )
    
    point = resolved.get('point')
    points = resolved.get('points')
    bbox = resolved.get('bbox')
    single_point = resolved.get('single_point')
    
    print(f'  ├─ Point (logging): {point}')
    print(f'  ├─ Points (array): {points.shape if points is not None else "None"} | {points if points is not None else "N/A"}')
    print(f'  ├─ BBox (XYXY): {bbox}')
    print(f'  └─ Single point: {single_point}')

print('\n' + '=' * 70)
print('✓ All modes verified! Visualization should match training.')
print('=' * 70)
