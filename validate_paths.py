#!/usr/bin/env python3
"""
Validate prediction paths in results.csv against actual files on disk.
Useful for debugging "pred not found" issues in visualization pipeline.

Usage:
    python validate_paths.py --results outputs/phase1_xray_controlled/results.csv --config configs/phase1.yaml
    python validate_paths.py --exp_dir outputs/phase1_xray_controlled
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import yaml

from viz.path_resolver import (
    validate_paths_in_df,
    format_path_validation_report,
    resolve_pred_path,
    get_pred_root
)


def load_config(config_path: Path) -> dict:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Validate prediction paths in results.csv")
    parser.add_argument("--results", type=str, help="Path to results.csv")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--exp_dir", type=str, help="Experiment directory (alternative to --results + --config)")
    parser.add_argument("--noise_seed", type=int, default=42, help="Noise seed to check")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", type=str, help="Output report to file")
    parser.add_argument("--check_sample", type=str, help="Check specific sample ID")
    
    args = parser.parse_args()
    
    # Determine paths
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        results_path = exp_dir / "results.csv"
        config_path = exp_dir / "meta" / "config_snapshot.yaml"
        if not config_path.exists():
            config_path = exp_dir / "meta" / "config_snapshot.json"
    else:
        results_path = Path(args.results) if args.results else None
        config_path = Path(args.config) if args.config else None
    
    if not results_path or not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        sys.exit(1)
    
    if not config_path or not config_path.exists():
        print(f"[WARNING] Config file not found: {config_path}")
        # Create minimal config from results path
        exp_dir = results_path.parent
        cfg = {
            "exp": {
                "name": exp_dir.name,
                "out_root": str(exp_dir.parent)
            }
        }
    else:
        if str(config_path).endswith(".json"):
            import json
            with open(config_path) as f:
                cfg = json.load(f)
        else:
            cfg = load_config(config_path)
    
    print(f"[INFO] Loading results from: {results_path}")
    df = pd.read_csv(results_path)
    print(f"[INFO] Loaded {len(df)} rows")
    
    pred_root = get_pred_root(cfg)
    print(f"[INFO] Prediction root: {pred_root}")
    print(f"[INFO] Pred root exists: {pred_root.exists()}")
    
    if pred_root.exists():
        # List immediate subdirectories
        subdirs = list(pred_root.iterdir())
        print(f"[INFO] Subdirectories in pred_root: {[d.name for d in subdirs[:10]]}")
    
    # Check specific sample if requested
    if args.check_sample:
        print(f"\n[INFO] Checking specific sample: {args.check_sample}")
        sample_rows = df[df["id"] == args.check_sample]
        if len(sample_rows) == 0:
            print(f"[ERROR] Sample {args.check_sample} not found in results")
        else:
            for idx, row in sample_rows.iterrows():
                result = resolve_pred_path(
                    pred_root=pred_root,
                    dataset=row.get("dataset", ""),
                    model=row.get("model", ""),
                    weight=row.get("weight", ""),
                    mode=row.get("mode", ""),
                    protocol=row.get("protocol", "P0"),
                    noise=row.get("noise", "clean"),
                    level=str(row.get("level", "L0")),
                    sid=str(row.get("id", "")),
                    noise_seed=args.noise_seed,
                    log_debug=True
                )
                print(f"\n  Mode: {row.get('mode')}, Noise: {row.get('noise')}, Level: {row.get('level')}")
                print(f"  Found: {result.found}")
                if result.found:
                    print(f"  Path: {result.path}")
                    print(f"  Strategy: {result.matched_pattern}")
                else:
                    print("  Search attempts:")
                    for attempt in result.search_attempts[:5]:
                        exists = Path(attempt).exists() if not attempt.startswith("glob") else "N/A"
                        print(f"    - {attempt} (exists: {exists})")
        return
    
    # Full validation
    print("\n[INFO] Validating all paths...")
    stats = validate_paths_in_df(
        df, 
        cfg, 
        noise_seed=args.noise_seed,
        verbose=args.verbose
    )
    
    report = format_path_validation_report(stats)
    print(report)
    
    # Detailed analysis
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)
    
    # Check for case sensitivity issues
    datasets_in_df = df["dataset"].unique()
    print(f"\nDatasets in CSV: {list(datasets_in_df)}")
    
    if pred_root.exists():
        datasets_on_disk = [d.name for d in pred_root.iterdir() if d.is_dir()]
        print(f"Datasets on disk: {datasets_on_disk}")
        
        for ds in datasets_in_df:
            if ds not in datasets_on_disk:
                # Check for case variant
                for disk_ds in datasets_on_disk:
                    if disk_ds.lower() == ds.lower():
                        print(f"\n[ISSUE] Case mismatch: CSV has '{ds}', disk has '{disk_ds}'")
    
    # Check for seed folder pattern
    print("\n[INFO] Checking seed folder patterns...")
    sample_row = df.iloc[0]
    base_path = pred_root / sample_row["dataset"] / sample_row["model"] / sample_row["weight"] / sample_row["mode"]
    if base_path.exists():
        # Walk to find actual file structure
        for protocol_dir in base_path.iterdir():
            if protocol_dir.is_dir():
                for noise_dir in protocol_dir.iterdir():
                    if noise_dir.is_dir():
                        for level_dir in noise_dir.iterdir():
                            if level_dir.is_dir():
                                contents = list(level_dir.iterdir())
                                has_seed_folder = any(c.is_dir() and c.name.startswith("seed") for c in contents)
                                has_direct_files = any(c.is_file() and c.suffix in [".png", ".jpg"] for c in contents)
                                print(f"  {level_dir.relative_to(pred_root)}: seed_folder={has_seed_folder}, direct_files={has_direct_files}")
                                # Only show first few
                                break
                        break
                break
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[INFO] Report saved to: {output_path}")
    
    # Suggest fixes
    print("\n" + "=" * 60)
    print("SUGGESTED FIXES")
    print("=" * 60)
    
    if stats["not_found"] > 0:
        print("""
1. Case sensitivity: Ensure dataset name in CSV matches folder name exactly.
   
2. Seed folder: The experiment saves predictions to 'L0/seed42/' but visualization
   may be looking in 'L0/' directly. The path_resolver.py handles both cases.
   
3. To fix existing code, import and use resolve_pred_path() instead of building
   paths manually:
   
   from viz.path_resolver import resolve_pred_path, get_pred_root
   
   pred_root = get_pred_root(cfg)
   result = resolve_pred_path(
       pred_root, dataset, model, weight, mode, 
       protocol, noise, level, sid, noise_seed=42
   )
   if result.found:
       pred = _safe_read_mask(str(result.path))
   else:
       print(f"Not found: {result.search_attempts[0]}")
""")


if __name__ == "__main__":
    main()
