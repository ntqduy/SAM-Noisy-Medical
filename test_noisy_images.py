#!/usr/bin/env python
"""
Test helper script for noisy images saving and visualization pipeline.

This script tests:
1. Loading results.csv and picking sample IDs
2. Verifying noisy_images paths exist for P1 levels
3. Exporting preview.pdf and multilevel failure PDF
4. Asserting files are created correctly

Usage:
    python test_noisy_images.py --exp_dir outputs/phase1_xray_controlled [--num_samples 3]
"""
import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_config(exp_dir: Path) -> dict:
    """Load configuration from experiment directory."""
    config_path = exp_dir / "meta" / "config_snapshot.yaml"
    if not config_path.exists():
        config_path = exp_dir / "meta" / "config_snapshot.json"
    
    if config_path.exists():
        if config_path.suffix == ".yaml":
            with open(config_path) as f:
                return yaml.safe_load(f)
        else:
            import json
            with open(config_path) as f:
                return json.load(f)
    else:
        # Return minimal config
        return {
            "exp": {"name": exp_dir.name, "out_root": str(exp_dir.parent)},
            "noise_config": {"base_seed": 42},
            "levels": {"names": ["L0", "L1", "L2", "L3", "L4"]}
        }


def test_noisy_image_paths(exp_dir: Path, num_samples: int = 3) -> dict:
    """
    Test 1: Verify noisy_images paths exist for P1 levels.
    
    Args:
        exp_dir: Experiment output directory
        num_samples: Number of sample IDs to check
        
    Returns:
        Dict with test results
    """
    logger.info("=" * 60)
    logger.info("TEST 1: Verify noisy_images paths for P1 levels")
    logger.info("=" * 60)
    
    results = {
        "passed": False,
        "samples_checked": 0,
        "paths_found": 0,
        "paths_missing": 0,
        "details": []
    }
    
    results_csv = exp_dir / "results.csv"
    if not results_csv.exists():
        logger.error(f"results.csv not found: {results_csv}")
        return results
    
    df = pd.read_csv(results_csv)
    cfg = load_config(exp_dir)
    
    # Import path resolver
    sys.path.insert(0, str(exp_dir.parent.parent))
    from viz.path_resolver import resolve_noisy_image_path, get_noisy_root
    
    noisy_root = get_noisy_root(cfg)
    noise_seed = cfg.get("noise_config", {}).get("base_seed", 42)
    
    logger.info(f"Noisy images root: {noisy_root}")
    logger.info(f"Noise seed: {noise_seed}")
    
    # Get sample IDs from P0 baseline
    base = df[df["protocol"] == "P0"]
    sample_ids = base["id"].dropna().unique().tolist()[:num_samples]
    
    logger.info(f"Checking {len(sample_ids)} samples: {sample_ids}")
    
    # Get noise types
    noises = [n for n in df["noise"].unique() if n != "clean"]
    levels = ["L1", "L2", "L3", "L4"]  # P1 levels
    
    for sid in sample_ids:
        results["samples_checked"] += 1
        
        # Get dataset from row
        row = base[base["id"] == sid].iloc[0]
        dataset = row["dataset"]
        
        for noise in noises[:2]:  # Check first 2 noise types
            for lv in levels:
                path_result = resolve_noisy_image_path(
                    noisy_root=noisy_root,
                    dataset=dataset,
                    noise=noise,
                    level=lv,
                    sid=str(sid),
                    noise_seed=noise_seed,
                    log_debug=False
                )
                
                if path_result.found:
                    results["paths_found"] += 1
                    results["details"].append({
                        "sid": sid, "noise": noise, "level": lv,
                        "found": True, "path": str(path_result.path)
                    })
                else:
                    results["paths_missing"] += 1
                    results["details"].append({
                        "sid": sid, "noise": noise, "level": lv,
                        "found": False, "searched": path_result.search_attempts[0] if path_result.search_attempts else "N/A"
                    })
    
    total_checked = results["paths_found"] + results["paths_missing"]
    results["passed"] = results["paths_missing"] == 0 and results["paths_found"] > 0
    
    logger.info(f"Results: {results['paths_found']}/{total_checked} paths found")
    
    if results["paths_missing"] > 0:
        logger.warning(f"Missing paths: {results['paths_missing']}")
        for d in results["details"]:
            if not d["found"]:
                logger.warning(f"  [{d['sid']}/{d['noise']}/{d['level']}] Not found: {d.get('searched', 'N/A')}")
    
    if results["paths_found"] == 0:
        logger.info("NOTE: Noisy images may not exist yet. Run experiment with outputs.save_noisy_images=True")
    
    return results


def test_preview_pdf_export(exp_dir: Path, num_samples: int = 3) -> dict:
    """
    Test 2: Export preview.pdf and verify it's created.
    
    Args:
        exp_dir: Experiment output directory
        num_samples: Number of samples to include
        
    Returns:
        Dict with test results
    """
    logger.info("=" * 60)
    logger.info("TEST 2: Export preview.pdf")
    logger.info("=" * 60)
    
    results = {
        "passed": False,
        "output_path": None,
        "error": None
    }
    
    results_csv = exp_dir / "results.csv"
    if not results_csv.exists():
        results["error"] = f"results.csv not found: {results_csv}"
        logger.error(results["error"])
        return results
    
    df = pd.read_csv(results_csv)
    cfg = load_config(exp_dir)
    
    # Import visualization module
    sys.path.insert(0, str(exp_dir.parent.parent))
    from viz.grids import save_preview_pdf
    
    # Output path
    test_output_dir = exp_dir / "test_outputs"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = test_output_dir / "test_preview.pdf"
    
    try:
        save_preview_pdf(
            df=df,
            cfg=cfg,
            out_pdf=out_pdf,
            num_samples=num_samples,
            levels=["L0", "L2", "L4"]
        )
        
        if out_pdf.exists():
            results["passed"] = True
            results["output_path"] = str(out_pdf)
            logger.info(f"[PASS] preview.pdf created: {out_pdf}")
            logger.info(f"  File size: {out_pdf.stat().st_size / 1024:.1f} KB")
        else:
            results["error"] = "PDF was not created"
            logger.error(f"[FAIL] PDF not created at {out_pdf}")
            
    except Exception as e:
        results["error"] = str(e)
        logger.error(f"[FAIL] Error creating preview.pdf: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def test_multilevel_failure_export(exp_dir: Path, num_samples: int = 2) -> dict:
    """
    Test 3: Export multilevel failure PDF and verify it's created.
    
    Args:
        exp_dir: Experiment output directory
        num_samples: Number of failure cases to export
        
    Returns:
        Dict with test results
    """
    logger.info("=" * 60)
    logger.info("TEST 3: Export multilevel failure cases")
    logger.info("=" * 60)
    
    results = {
        "passed": False,
        "output_paths": [],
        "error": None
    }
    
    results_csv = exp_dir / "results.csv"
    if not results_csv.exists():
        results["error"] = f"results.csv not found: {results_csv}"
        logger.error(results["error"])
        return results
    
    df = pd.read_csv(results_csv)
    cfg = load_config(exp_dir)
    
    # Import visualization module
    sys.path.insert(0, str(exp_dir.parent.parent))
    from viz.failure_cases_v2 import export_failure_cases_multilevel
    
    # Output directory
    test_output_dir = exp_dir / "test_outputs" / "failure_cases"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    noise_seed = cfg.get("noise_config", {}).get("base_seed", 42)
    
    try:
        exported = export_failure_cases_multilevel(
            df=df,
            cfg=cfg,
            out_dir=test_output_dir,
            top_k_fail=num_samples,
            levels=["L0", "L1", "L2", "L3", "L4"],
            per_mode=False,
            per_noise=False,
            metric="dice",
            noise_seed=noise_seed
        )
        
        if len(exported) > 0:
            results["passed"] = True
            results["output_paths"] = exported
            logger.info(f"[PASS] Exported {len(exported)} failure case PDFs")
            for p in exported[:3]:
                logger.info(f"  - {p}")
        else:
            results["error"] = "No PDFs were exported"
            logger.warning("[WARN] No failure case PDFs created (may have no failures)")
            
    except Exception as e:
        results["error"] = str(e)
        logger.error(f"[FAIL] Error creating multilevel failure PDFs: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def run_all_tests(exp_dir: Path, num_samples: int = 3) -> dict:
    """Run all tests and return summary."""
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING ALL NOISY IMAGE VISUALIZATION TESTS")
    logger.info("=" * 70 + "\n")
    
    all_results = {}
    
    # Test 1: Noisy image paths
    all_results["noisy_paths"] = test_noisy_image_paths(exp_dir, num_samples)
    logger.info("")
    
    # Test 2: Preview PDF
    all_results["preview_pdf"] = test_preview_pdf_export(exp_dir, num_samples)
    logger.info("")
    
    # Test 3: Multilevel failure
    all_results["multilevel_failure"] = test_multilevel_failure_export(exp_dir, num_samples)
    logger.info("")
    
    # Summary
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    total_passed = 0
    total_tests = 0
    
    for test_name, result in all_results.items():
        total_tests += 1
        status = "[PASS]" if result.get("passed") else "[FAIL]"
        if result.get("passed"):
            total_passed += 1
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    # Special notes
    if all_results["noisy_paths"]["paths_found"] == 0:
        logger.info("\nNOTE: No noisy images found. This is expected if:")
        logger.info("  1. Experiment was run before this feature was added")
        logger.info("  2. outputs.save_noisy_images was set to False")
        logger.info("  To generate noisy images, re-run the experiment with:")
        logger.info("    outputs.save_noisy_images: true")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Test noisy images visualization pipeline")
    parser.add_argument("--exp_dir", type=str, default="outputs/phase1_xray_controlled",
                       help="Experiment output directory")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples to test")
    
    args = parser.parse_args()
    
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        logger.error(f"Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    results = run_all_tests(exp_dir, args.num_samples)
    
    # Exit with error code if any test failed
    all_passed = all(r.get("passed", False) for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
