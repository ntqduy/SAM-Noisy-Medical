"""
Severity Mapping Direction Documentation for NoisySAM Benchmark
===============================================================

This document explains how each noise type maps its primary parameter to severity.
Understanding this is critical for correct severity normalization, ranking, and
robustness analysis.

TWO MAPPING TYPES
-----------------

1. STANDARD MAPPING (higher parameter = higher severity = more degradation)
   - As the parameter increases, the image quality decreases
   - severity_scalar = (param - min) / (max - min)

2. INVERTED MAPPING (lower parameter = higher severity = more degradation)
   - As the parameter decreases, the image quality decreases
   - severity_scalar = 1.0 - (param - min) / (max - min)


NOISE-BY-NOISE BREAKDOWN
------------------------

STANDARD MAPPING NOISES (higher param = more degradation):
----------------------------------------------------------

| Noise Type        | Primary Param | Mild Value | Severe Value | Rationale                          |
|-------------------|---------------|------------|--------------|-------------------------------------|
| GaussianNoise     | sigma         | 5          | 80           | Higher σ = more additive noise      |
| SaltPepperNoise   | amount        | 0.01       | 0.50         | Higher amount = more corrupted pixels|
| MotionBlur        | k             | 3          | 40           | Larger kernel = more blur           |
| SpeckleNoise      | sigma         | 0.05       | 0.90         | Higher σ = more multiplicative noise|
| RicianNoise       | sigma         | 5          | 75           | Higher σ = more MRI noise           |
| PixelationNoise   | block_size    | 2          | 50           | Larger blocks = more pixelation     |
| HighBrightnessNoise| factor       | 1.05       | 5.0          | Higher factor = more overexposure   |
| HighContrastNoise | factor        | 1.5        | 8.0          | Higher factor = more clipping       |
| BiasField         | strength      | 0.08       | 1.25         | Higher strength = more inhomogeneity|
| CoarseDropout     | holes/size    | 1/8        | 20/64        | More/larger holes = more dropout    |
| GridMask          | r             | 8          | 48           | Larger mask width = more occlusion  |
| QuantizationNoise | step          | 1          | 64           | Larger step = more posterization    |
| DefocusBlur       | k             | 3          | 31           | Larger kernel = more blur           |
| UniformNoise      | b             | 10         | 50           | Larger bound = more noise           |


INVERTED MAPPING NOISES (lower param = more degradation):
---------------------------------------------------------

| Noise Type        | Primary Param | Mild Value | Severe Value | Rationale                          |
|-------------------|---------------|------------|--------------|-------------------------------------|
| PoissonNoise      | peak (lam)    | 80         | 4            | Lower peak = fewer photons = more noise|
| JPEGArtifacts     | quality       | 90         | 5            | Lower quality = more compression artifacts|
| LowContrastNoise  | alpha         | 0.95       | 0.1          | Lower α = image pushed toward gray  |
| LowBrightnessNoise| factor        | 0.95       | 0.05         | Lower factor = darker = less visible|


CODE IMPLEMENTATION NOTES
-------------------------

For STANDARD mapping noises, the base class default works:
```python
def get_severity_scalar(self) -> float:
    normalized = (val - min_val) / (max_val - min_val)
    return float(np.clip(normalized, 0.0, 1.0))
```

For INVERTED mapping noises, must override:
```python
def get_severity_scalar(self) -> float:
    normalized = 1.0 - (val - min_val) / (max_val - min_val)  # Note: 1.0 - ...
    return float(np.clip(normalized, 0.0, 1.0))
```


CONFIG LEVEL PROGRESSION
------------------------

All noise configs in full_benchmark.yaml follow:
- L1 = mildest (lowest severity_scalar)
- L9 = most severe (highest severity_scalar)

For STANDARD noises: L1 has smallest param, L9 has largest param
For INVERTED noises: L1 has largest param, L9 has smallest param

Example - Gaussian (STANDARD):
  L1: sigma=5   → severity = (5-0)/(90-0)   = 0.056
  L9: sigma=80  → severity = (80-0)/(90-0)  = 0.889

Example - JPEG (INVERTED):
  L1: quality=90 → severity = 1 - (90-5)/(95-5) = 0.056
  L9: quality=5  → severity = 1 - (5-5)/(95-5)  = 1.000

Example - Poisson (INVERTED):
  L1: peak=80   → severity = 1 - (80-1)/(100-1) = 0.202
  L9: peak=4    → severity = 1 - (4-1)/(100-1)  = 0.970


METRICS NOTE
------------

Severity mapping is about noise parameters, NOT about metric interpretation.

For metric interpretation in robustness analysis:
- IoU, Dice, Recall, Precision, F1: HIGHER is BETTER
- HD (Hausdorff Distance): LOWER is BETTER

These are separate concerns:
- severity_scalar: how much noise was applied (0=clean, 1=max noise)
- metric_value: how well the model performed (depends on metric direction)


ADDING NEW NOISES
-----------------

When implementing a new noise class:

1. Determine if higher param = more degradation (STANDARD) or less (INVERTED)

2. If STANDARD: use base class default get_severity_scalar()

3. If INVERTED: override get_severity_scalar() with 1.0 - normalization

4. Document the mapping in this file

5. Ensure PARAM_RANGES covers the full range used in configs

6. Test:
   - L1 should give low severity_scalar
   - L9 should give high severity_scalar
   - Visual inspection: L9 should look more degraded than L1
"""

# Type annotations for programmatic use
from typing import Dict, Literal

SeverityMapping = Literal["standard", "inverted"]

# Registry of severity mapping directions
SEVERITY_MAPPING_DIRECTION: Dict[str, SeverityMapping] = {
    # Standard mapping (higher param = higher severity)
    "gaussian": "standard",
    "salt_pepper": "standard",
    "motion_blur": "standard",
    "speckle": "standard",
    "rician": "standard",
    "pixelation": "standard",
    "high_brightness": "standard",
    "high_contrast": "standard",
    "bias_field": "standard",
    "coarse_dropout": "standard",
    "gridmask": "standard",
    "quantization": "standard",
    "defocus_blur": "standard",
    "uniform": "standard",

    # Inverted mapping (lower param = higher severity)
    "poisson": "inverted",
    "jpeg": "inverted",
    "low_contrast": "inverted",
    "low_brightness": "inverted",

    # Special
    "clean": "standard",  # Always severity = 0
}


def get_severity_direction(noise_type: str) -> SeverityMapping:
    """
    Get the severity mapping direction for a noise type.

    Parameters
    ----------
    noise_type : str
        Name of the noise type (e.g., 'gaussian', 'poisson')

    Returns
    -------
    SeverityMapping
        'standard' if higher param = higher severity
        'inverted' if lower param = higher severity
    """
    return SEVERITY_MAPPING_DIRECTION.get(noise_type.lower(), "standard")


def is_inverted_severity(noise_type: str) -> bool:
    """
    Check if a noise type uses inverted severity mapping.

    Parameters
    ----------
    noise_type : str
        Name of the noise type

    Returns
    -------
    bool
        True if lower parameter value means higher severity
    """
    return get_severity_direction(noise_type) == "inverted"
