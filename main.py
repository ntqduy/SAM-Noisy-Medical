"""
Main entry point for AIO25 NoisySAM benchmark.

Extended CLI options:
  --config       Config file path (required)
  --phase        Override phase (1 or 2)
  --dry_run      Print protocol cases without running
  --limit_n      Limit number of samples
  --use_cache    Enable prediction caching
  --clear_cache  Clear existing cache before running
  --only_report  Skip inference, generate report from existing results
  --max_samples  Limit samples per protocol case
  --max_noise    Filter to specific noise types
  --max_level    Filter to specific levels
  --debug        Enable debug mode (implies --use_cache)
"""
import argparse
from pathlib import Path

from runner.experiment import run_experiment
from runner.io_utils import load_yaml_config
from runner.config_schema import apply_cli_overrides


def parse_args():
    ap = argparse.ArgumentParser(
        description="AIO25 NoisySAM Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark
  python main.py --config configs/phase1.yaml
  
  # Quick debug run with caching
  python main.py --config configs/phase1.yaml --debug --max_samples 5
  
  # Re-generate report from existing results
  python main.py --config configs/phase1.yaml --only_report
  
  # Run specific noise types only
  python main.py --config configs/phase1.yaml --max_noise gaussian,poisson
  
  # Clear cache and rerun
  python main.py --config configs/phase1.yaml --clear_cache --use_cache
"""
    )
    
    # Required
    ap.add_argument("--config", type=str, required=True, 
                    help="Path to config file (e.g., configs/phase1.yaml)")
    
    # Phase override
    ap.add_argument("--phase", type=int, choices=[1, 2], default=None,
                    help="Override phase (1 or 2)")
    
    # Run modes
    ap.add_argument("--dry_run", action="store_true",
                    help="Print protocol cases without running inference")
    ap.add_argument("--only_report", action="store_true",
                    help="Skip inference, generate report from existing results.csv")
    
    # Sampling limits
    ap.add_argument("--limit_n", type=int, default=None,
                    help="Limit total number of samples (deprecated, use --max_samples)")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Maximum samples per protocol case")
    
    # Cache options
    ap.add_argument("--use_cache", action="store_true",
                    help="Enable prediction caching to skip repeated inferences")
    ap.add_argument("--clear_cache", action="store_true",
                    help="Clear existing cache before running")
    
    # Debug filters
    ap.add_argument("--debug", action="store_true",
                    help="Enable debug mode (implies --use_cache, single model/noise)")
    ap.add_argument("--max_noise", type=str, default=None,
                    help="Comma-separated list of noise types to run (e.g., gaussian,poisson)")
    ap.add_argument("--max_level", type=str, default=None,
                    help="Comma-separated list of levels to run (e.g., L1,L2)")
    ap.add_argument("--max_models", type=int, default=None,
                    help="Limit number of models to test")
    
    # Output options
    ap.add_argument("--out_root", type=str, default=None,
                    help="Override output root directory")
    ap.add_argument("--exp_name", type=str, default=None,
                    help="Override experiment name")
    
    # Verbosity
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable verbose output")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="Suppress progress output")
    
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(Path(args.config))

    # Apply basic overrides
    if args.phase is not None:
        cfg["phase"] = int(args.phase)
    if args.dry_run:
        cfg["dry_run"] = True
    if args.limit_n is not None:
        cfg["limit_n"] = int(args.limit_n)
    if args.out_root is not None:
        cfg["exp"]["out_root"] = args.out_root
    if args.exp_name is not None:
        cfg["exp"]["name"] = args.exp_name
    
    # Apply extended CLI overrides
    cfg = apply_cli_overrides(cfg, vars(args))
    
    # Handle debug mode
    if args.debug:
        cfg["cache"] = cfg.get("cache", {})
        cfg["cache"]["enabled"] = True
        cfg["debug"] = cfg.get("debug", {})
        cfg["debug"]["enabled"] = True
        if args.verbose:
            print("[DEBUG] Debug mode enabled with caching")
    
    # Parse noise/level filters
    if args.max_noise:
        cfg["debug"] = cfg.get("debug", {})
        cfg["debug"]["noise_types"] = [n.strip() for n in args.max_noise.split(",")]
    
    if args.max_level:
        cfg["debug"] = cfg.get("debug", {})
        cfg["debug"]["levels"] = [lv.strip() for lv in args.max_level.split(",")]
    
    if args.max_samples:
        cfg["debug"] = cfg.get("debug", {})
        cfg["debug"]["max_samples"] = args.max_samples
    
    if args.max_models:
        cfg["debug"] = cfg.get("debug", {})
        cfg["debug"]["max_models"] = args.max_models
    
    # Cache settings
    if args.use_cache:
        cfg["cache"] = cfg.get("cache", {})
        cfg["cache"]["enabled"] = True
    
    if args.clear_cache:
        cfg["cache"] = cfg.get("cache", {})
        cfg["cache"]["clear_on_start"] = True
    
    # Report-only mode
    if args.only_report:
        cfg["only_report"] = True
    
    # Verbosity
    if args.verbose:
        cfg["verbose"] = True
    if args.quiet:
        cfg["verbose"] = False
    
    run_experiment(cfg)


if __name__ == "__main__":
    main()
