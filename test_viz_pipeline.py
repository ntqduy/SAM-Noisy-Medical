#!/usr/bin/env python3
"""
Test script for the updated visualization pipeline.
Verifies that path resolution and failure case export work correctly.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yaml

def test_path_resolver():
    """Test path resolver functions."""
    print("\n" + "=" * 60)
    print("TEST 1: Path Resolver")
    print("=" * 60)
    
    from viz.path_resolver import (
        resolve_pred_path,
        get_pred_root,
        validate_paths_in_df,
        format_path_validation_report
    )
    
    # Load config
    config_path = Path("outputs/phase1_xray_controlled/meta/config_snapshot.yaml")
    if not config_path.exists():
        print(f"[SKIP] Config not found: {config_path}")
        return
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    pred_root = get_pred_root(cfg)
    print(f"Pred root: {pred_root}")
    print(f"Exists: {pred_root.exists()}")
    
    # Test single path resolution
    result = resolve_pred_path(
        pred_root=pred_root,
        dataset="montgomery",
        model="SAM",
        weight="sam_b",
        mode="prompt_bbox",
        protocol="P0",
        noise="clean",
        level="L0",
        sid="MCUCXR_0001_0",
        noise_seed=42,
        log_debug=True
    )
    
    print(f"\nTest resolve_pred_path:")
    print(f"  Found: {result.found}")
    print(f"  Path: {result.path}")
    print(f"  Strategy: {result.matched_pattern}")
    
    assert result.found, "Path should be found!"
    print("\n[PASS] Path resolver test PASSED")


def test_failure_cases_export():
    """Test failure cases export with new path resolution."""
    print("\n" + "=" * 60)
    print("TEST 2: Failure Cases Export")
    print("=" * 60)
    
    from viz.failure_cases import (
        identify_top_failures,
        export_failure_cases
    )
    
    # Load data
    results_path = Path("outputs/phase1_xray_controlled/results.csv")
    config_path = Path("outputs/phase1_xray_controlled/meta/config_snapshot.yaml")
    
    if not results_path.exists():
        print(f"[SKIP] Results not found: {results_path}")
        return
    
    df = pd.read_csv(results_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Test identify_top_failures
    failures = identify_top_failures(df, top_k=5, metric="dice")
    print(f"\nTop failures identified: {len(failures)}")
    if len(failures) > 0:
        print(failures[["id", "mode", "noise", "dice_L0", "dice_L4", "drop"]].head())
    
    # Test export (dry run - just test path resolution)
    test_out_dir = Path("outputs/phase1_xray_controlled/test_failure_export")
    print(f"\nExporting to: {test_out_dir}")
    
    exported = export_failure_cases(
        df, cfg, test_out_dir, top_k=3, metric="dice"
    )
    
    print(f"Exported files: {len(exported)}")
    for f in exported:
        print(f"  - {Path(f).name}")
    
    # Verify files exist
    for f in exported:
        assert Path(f).exists(), f"Exported file should exist: {f}"
    
    print("\n✓ Failure cases export test PASSED")


def test_multilevel_export():
    """Test multi-level failure cases export."""
    print("\n" + "=" * 60)
    print("TEST 3: Multi-Level Failure Cases Export")
    print("=" * 60)
    
    try:
        from viz.failure_cases_v2 import (
            export_failure_cases_multilevel,
            export_random_cases_multilevel
        )
    except ImportError as e:
        print(f"[SKIP] Multi-level export not available: {e}")
        return
    
    # Load data
    results_path = Path("outputs/phase1_xray_controlled/results.csv")
    config_path = Path("outputs/phase1_xray_controlled/meta/config_snapshot.yaml")
    
    if not results_path.exists():
        print(f"[SKIP] Results not found: {results_path}")
        return
    
    df = pd.read_csv(results_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Test multi-level export
    test_out_dir = Path("outputs/phase1_xray_controlled/test_multilevel_export")
    print(f"\nExporting multi-level to: {test_out_dir}")
    
    exported = export_failure_cases_multilevel(
        df, cfg, test_out_dir,
        top_k_fail=2,
        levels=["L0", "L2", "L4"],
        per_mode=True,
        per_noise=False
    )
    
    print(f"Exported files: {len(exported)}")
    for f in exported[:5]:
        print(f"  - {Path(f).name}")
    
    print("\n✓ Multi-level export test PASSED")


def test_grids_path_resolution():
    """Test grids.py path resolution."""
    print("\n" + "=" * 60)
    print("TEST 4: Grids Path Resolution")
    print("=" * 60)
    
    from viz.grids import _resolve_pred_for_row
    
    # Load data
    results_path = Path("outputs/phase1_xray_controlled/results.csv")
    config_path = Path("outputs/phase1_xray_controlled/meta/config_snapshot.yaml")
    
    if not results_path.exists():
        print(f"[SKIP] Results not found: {results_path}")
        return
    
    df = pd.read_csv(results_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Test with first row
    row = df.iloc[0]
    result = _resolve_pred_for_row(row, cfg, row["id"], noise_seed=42, log_debug=True)
    
    print(f"\nTest _resolve_pred_for_row:")
    print(f"  Sample: {row['id']}")
    print(f"  Found: {result.found}")
    print(f"  Path: {result.path}")
    
    assert result.found, "Should find prediction path"
    print("\n✓ Grids path resolution test PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("NoisySAM Visualization Pipeline Test Suite")
    print("=" * 60)
    
    tests = [
        test_path_resolver,
        test_failure_cases_export,
        test_multilevel_export,
        test_grids_path_resolution
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {e}")
            failed += 1
        except Exception as e:
            if "[SKIP]" in str(e):
                skipped += 1
            else:
                print(f"\n✗ TEST ERROR: {e}")
                failed += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print("=" * 60)
