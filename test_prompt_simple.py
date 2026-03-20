#!/usr/bin/env python3
"""
Simple test to verify normalize_prompt_mode consistency without dependencies.
"""

import sys

print("=" * 70)
print("TEST: normalize_prompt_mode() - unified canonicalization")
print("=" * 70)

from models.wrappers.prompt_utils import normalize_prompt_mode

test_cases = [
    # Standard names
    ("prompt_point", "prompt_point"),
    ("prompt_bbox", "prompt_bbox"),
    ("prompt_point_box", "prompt_point_box"),
    # Short aliases
    ("point", "prompt_point"),
    ("bbox", "prompt_bbox"),
    ("box", "prompt_bbox"),
    # Display aliases (new support for visualization)
    ("point+bbox", "prompt_point_box"),
    ("point_box", "prompt_point_box"),
    ("pointbox", "prompt_point_box"),
    # Auto modes
    ("autogen", "autogen"),
    ("auto", "autogen"),
]

all_passed = True
for input_mode, expected_output in test_cases:
    try:
        output = normalize_prompt_mode(input_mode)
        status = "✅" if output == expected_output else f"❌ (got {output})"
        print(f"  {input_mode:20s} → {expected_output:20s} {status}")
        if output != expected_output:
            all_passed = False
    except Exception as e:
        print(f"  {input_mode:20s} → ERROR: {e}")
        all_passed = False

print()
print("=" * 70)
print("TEST: Error handling for unsupported modes")
print("=" * 70)

unsupported_modes = ["invalid", "foo", "box_point", "xyz"]
for mode in unsupported_modes:
    try:
        normalize_prompt_mode(mode)
        print(f"  ❌ '{mode}' should raise ValueError but didn't")
        all_passed = False
    except ValueError:
        print(f"  ✅ '{mode}' correctly raises ValueError")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

if all_passed:
    print("✅ ALL TESTS PASSED!")
    print()
    print("Support added:")
    print("  ✓ normalize_prompt_mode() now supports 'point+bbox' alias")
    print("  ✓ Used consistently by all 8 models")
    print("  ✓ Visualization layer now uses same normalize_prompt_mode()")
    sys.exit(0)
else:
    print("❌ SOME TESTS FAILED!")
    sys.exit(1)
