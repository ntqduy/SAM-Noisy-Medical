"""
Path resolution utilities for NoisySAM visualization pipeline.
Handles multiple path formats with fallback search and debug logging.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import glob

# Configure logging
logger = logging.getLogger(__name__)


class PathResolutionResult:
    """Result of path resolution attempt."""
    
    def __init__(
        self,
        found: bool,
        path: Optional[Path],
        search_attempts: List[str],
        matched_pattern: Optional[str] = None
    ):
        self.found = found
        self.path = path
        self.search_attempts = search_attempts
        self.matched_pattern = matched_pattern
    
    def __repr__(self) -> str:
        return f"PathResolutionResult(found={self.found}, path={self.path})"


def resolve_pred_path(
    pred_root: Path,
    dataset: str,
    model: str,
    weight: str,
    mode: str,
    protocol: str,
    noise: str,
    level: str,
    sid: str,
    noise_seed: int = 42,
    extensions: List[str] = None,
    fallback_glob: bool = True,
    log_debug: bool = True
) -> PathResolutionResult:
    """
    Robustly resolve prediction mask path with multiple fallback strategies.
    
    Handles:
    - Case sensitivity (montgomery vs Montgomery)
    - seed subfolder (L4/ vs L4/seed42/)
    - Different extensions (.png, .jpg, .npy)
    - Zero-padded vs non-padded SIDs
    
    Args:
        pred_root: Base prediction directory
        dataset: Dataset name
        model: Model name
        weight: Weight ID
        mode: Inference mode (prompt_bbox, automatic)
        protocol: Protocol (P0, P1, P2)
        noise: Noise type (clean, gaussian, etc.)
        level: Level (L0, L1, L2, L3, L4)
        sid: Sample ID
        noise_seed: Noise seed for subfolder
        extensions: File extensions to try
        fallback_glob: Whether to use glob search as fallback
        log_debug: Whether to log debug info
        
    Returns:
        PathResolutionResult with found status and path
    """
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".npy"]
    
    search_attempts = []
    pred_root = Path(pred_root)
    
    # Strategy 1: Direct path (new structure with seed folder)
    for ext in extensions:
        path1 = pred_root / dataset / model / weight / mode / protocol / noise / level / f"seed{noise_seed}" / f"{sid}{ext}"
        search_attempts.append(str(path1))
        if path1.exists():
            if log_debug:
                logger.debug(f"[FOUND] Strategy 1 (with seed folder): {path1}")
            return PathResolutionResult(True, path1, search_attempts, "with_seed_folder")
    
    # Strategy 2: Direct path (old structure without seed folder)
    for ext in extensions:
        path2 = pred_root / dataset / model / weight / mode / protocol / noise / level / f"{sid}{ext}"
        search_attempts.append(str(path2))
        if path2.exists():
            if log_debug:
                logger.debug(f"[FOUND] Strategy 2 (without seed folder): {path2}")
            return PathResolutionResult(True, path2, search_attempts, "no_seed_folder")
    
    # Strategy 3: Case-insensitive dataset name
    for ds_variant in [dataset.lower(), dataset.upper(), dataset.title()]:
        if ds_variant == dataset:
            continue
        for ext in extensions:
            path3 = pred_root / ds_variant / model / weight / mode / protocol / noise / level / f"{sid}{ext}"
            search_attempts.append(str(path3))
            if path3.exists():
                if log_debug:
                    logger.debug(f"[FOUND] Strategy 3 (case variant: {ds_variant}): {path3}")
                return PathResolutionResult(True, path3, search_attempts, f"case_variant_{ds_variant}")
            
            # Also try with seed folder
            path3s = pred_root / ds_variant / model / weight / mode / protocol / noise / level / f"seed{noise_seed}" / f"{sid}{ext}"
            search_attempts.append(str(path3s))
            if path3s.exists():
                if log_debug:
                    logger.debug(f"[FOUND] Strategy 3s (case variant with seed): {path3s}")
                return PathResolutionResult(True, path3s, search_attempts, f"case_variant_with_seed_{ds_variant}")
    
    # Strategy 4: Try any seed folder
    for ext in extensions:
        pattern = str(pred_root / dataset / model / weight / mode / protocol / noise / level / "seed*" / f"{sid}{ext}")
        matches = glob.glob(pattern)
        if matches:
            path4 = Path(matches[0])
            if log_debug:
                logger.debug(f"[FOUND] Strategy 4 (any seed folder): {path4}")
            return PathResolutionResult(True, path4, search_attempts, "any_seed_folder")
    
    # Strategy 5: Glob fallback - search for file anywhere in level folder
    if fallback_glob:
        for ext in extensions:
            pattern = str(pred_root / "**" / model / weight / mode / protocol / noise / level / f"**/{sid}{ext}")
            search_attempts.append(f"glob: {pattern}")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                path5 = Path(matches[0])
                if log_debug:
                    logger.debug(f"[FOUND] Strategy 5 (glob fallback): {path5}")
                return PathResolutionResult(True, path5, search_attempts, "glob_fallback")
    
    # Not found
    if log_debug:
        logger.warning(f"[NOT FOUND] Pred mask for {sid} | {model}/{weight}/{mode}/{protocol}/{noise}/{level}")
        logger.debug(f"  Searched paths:\n" + "\n".join(f"    - {p}" for p in search_attempts[:6]))
    
    return PathResolutionResult(False, None, search_attempts, None)


def resolve_noisy_image_path(
    noisy_root: Path,
    dataset: str,
    noise: str,
    level: str,
    sid: str,
    noise_seed: int = 42,
    extensions: List[str] = None,
    fallback_glob: bool = True,
    log_debug: bool = False
) -> PathResolutionResult:
    """
    Resolve noisy image path (if saved during experiment).
    
    Structure: noisy_images/{dataset}/{noise}/{level}/seed{seed}/{sid}.png
    For L0/clean: noisy_images/{dataset}/clean/L0/seed{seed}/{sid}.png
    
    Args:
        noisy_root: Base noisy images directory
        dataset: Dataset name
        noise: Noise type (clean, gaussian, etc.)
        level: Level (L0, L1, L2, L3, L4)
        sid: Sample ID
        noise_seed: Noise seed for subfolder
        extensions: File extensions to try
        fallback_glob: Whether to use glob search as fallback
        log_debug: Whether to log debug info
        
    Returns:
        PathResolutionResult with found status and path
    """
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg"]
    
    search_attempts = []
    noisy_root = Path(noisy_root)
    
    # Strategy 1: With seed folder (new structure)
    for ext in extensions:
        path1 = noisy_root / dataset / noise / str(level) / f"seed{noise_seed}" / f"{sid}{ext}"
        search_attempts.append(str(path1))
        if path1.exists():
            if log_debug:
                logger.debug(f"[FOUND] Noisy image with seed folder: {path1}")
            return PathResolutionResult(True, path1, search_attempts, "noisy_with_seed")
    
    # Strategy 2: Without seed folder (old structure)
    for ext in extensions:
        path2 = noisy_root / dataset / noise / str(level) / f"{sid}{ext}"
        search_attempts.append(str(path2))
        if path2.exists():
            if log_debug:
                logger.debug(f"[FOUND] Noisy image without seed folder: {path2}")
            return PathResolutionResult(True, path2, search_attempts, "noisy_no_seed")
    
    # Strategy 3: Case-insensitive dataset name
    for ds_variant in [dataset.lower(), dataset.upper(), dataset.title()]:
        if ds_variant == dataset:
            continue
        for ext in extensions:
            path3 = noisy_root / ds_variant / noise / str(level) / f"seed{noise_seed}" / f"{sid}{ext}"
            search_attempts.append(str(path3))
            if path3.exists():
                if log_debug:
                    logger.debug(f"[FOUND] Noisy image case variant: {path3}")
                return PathResolutionResult(True, path3, search_attempts, f"noisy_case_{ds_variant}")
            
            path3b = noisy_root / ds_variant / noise / str(level) / f"{sid}{ext}"
            search_attempts.append(str(path3b))
            if path3b.exists():
                if log_debug:
                    logger.debug(f"[FOUND] Noisy image case variant (no seed): {path3b}")
                return PathResolutionResult(True, path3b, search_attempts, f"noisy_case_no_seed_{ds_variant}")
    
    # Strategy 4: Try any seed folder
    if fallback_glob:
        for ext in extensions:
            pattern = str(noisy_root / dataset / noise / str(level) / "seed*" / f"{sid}{ext}")
            matches = glob.glob(pattern)
            if matches:
                path4 = Path(matches[0])
                if log_debug:
                    logger.debug(f"[FOUND] Noisy image any seed folder: {path4}")
                return PathResolutionResult(True, path4, search_attempts, "noisy_any_seed")
    
    # Not found
    if log_debug:
        logger.debug(f"[NOT FOUND] Noisy image for {sid} | {noise}/{level}")
        logger.debug(f"  Top searched paths:\n" + "\n".join(f"    - {p}" for p in search_attempts[:4]))
    
    return PathResolutionResult(False, None, search_attempts, None)


def get_noisy_root(cfg: Dict[str, Any]) -> Path:
    """Get noisy images root directory from config."""
    exp_name = cfg["exp"]["name"]
    return Path(cfg["exp"].get("out_root", "outputs")) / exp_name / "noisy_images"


def should_save_noisy_images(cfg: Dict[str, Any]) -> bool:
    """Check if noisy images should be saved based on config."""
    outputs_cfg = cfg.get("outputs", {})
    return outputs_cfg.get("save_noisy_images", True)  # Default True for new experiments


def build_pred_path_candidates(
    cfg: Dict[str, Any],
    row: Dict[str, Any],
    level: Optional[str] = None,
    noise_seed: int = 42
) -> List[Path]:
    """
    Build list of candidate paths for a prediction mask based on config and row data.
    
    Args:
        cfg: Configuration dictionary
        row: DataFrame row (as dict) with dataset, model, weight, mode, protocol, noise, level, id
        level: Override level (optional)
        noise_seed: Noise seed
        
    Returns:
        List of candidate Path objects to try
    """
    exp_name = cfg["exp"]["name"]
    pred_root = Path(cfg["exp"].get("out_root", "outputs")) / exp_name / "pred_masks"
    
    dataset = row.get("dataset", "")
    model = row.get("model", "")
    weight = row.get("weight", "")
    mode = row.get("mode", "")
    protocol = row.get("protocol", "P0")
    noise = row.get("noise", "clean")
    lv = level or row.get("level", "L0")
    sid = row.get("id", "")
    
    candidates = []
    
    # Primary candidate
    candidates.append(pred_root / dataset / model / weight / mode / protocol / noise / str(lv) / f"seed{noise_seed}" / f"{sid}.png")
    candidates.append(pred_root / dataset / model / weight / mode / protocol / noise / str(lv) / f"{sid}.png")
    
    # Case variants
    for ds in [dataset.lower(), dataset.upper(), dataset.title()]:
        if ds != dataset:
            candidates.append(pred_root / ds / model / weight / mode / protocol / noise / str(lv) / f"seed{noise_seed}" / f"{sid}.png")
            candidates.append(pred_root / ds / model / weight / mode / protocol / noise / str(lv) / f"{sid}.png")
    
    return candidates


def get_pred_root(cfg: Dict[str, Any]) -> Path:
    """Get prediction root directory from config."""
    exp_name = cfg["exp"]["name"]
    return Path(cfg["exp"].get("out_root", "outputs")) / exp_name / "pred_masks"


def validate_paths_in_df(
    df, 
    cfg: Dict[str, Any],
    noise_seed: int = 42,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Validate prediction paths for all rows in DataFrame.
    
    Returns:
        Dict with statistics and missing paths
    """
    pred_root = get_pred_root(cfg)
    
    stats = {
        "total_rows": len(df),
        "found": 0,
        "not_found": 0,
        "found_by_strategy": {},
        "missing_paths": [],
        "found_by_mode": {},
        "found_by_noise": {},
        "found_by_level": {}
    }
    
    for idx, row in df.iterrows():
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
            noise_seed=noise_seed,
            log_debug=verbose
        )
        
        mode = row.get("mode", "unknown")
        noise = row.get("noise", "unknown")
        level = str(row.get("level", "unknown"))
        
        # Initialize counters
        for key in [mode, noise, level]:
            if mode not in stats["found_by_mode"]:
                stats["found_by_mode"][mode] = {"found": 0, "total": 0}
            if noise not in stats["found_by_noise"]:
                stats["found_by_noise"][noise] = {"found": 0, "total": 0}
            if level not in stats["found_by_level"]:
                stats["found_by_level"][level] = {"found": 0, "total": 0}
        
        stats["found_by_mode"][mode]["total"] += 1
        stats["found_by_noise"][noise]["total"] += 1
        stats["found_by_level"][level]["total"] += 1
        
        if result.found:
            stats["found"] += 1
            stats["found_by_mode"][mode]["found"] += 1
            stats["found_by_noise"][noise]["found"] += 1
            stats["found_by_level"][level]["found"] += 1
            
            pattern = result.matched_pattern or "unknown"
            stats["found_by_strategy"][pattern] = stats["found_by_strategy"].get(pattern, 0) + 1
        else:
            stats["not_found"] += 1
            if len(stats["missing_paths"]) < 50:  # Limit to first 50
                stats["missing_paths"].append({
                    "sid": row.get("id"),
                    "dataset": row.get("dataset"),
                    "model": row.get("model"),
                    "weight": row.get("weight"),
                    "mode": mode,
                    "protocol": row.get("protocol"),
                    "noise": noise,
                    "level": level,
                    "searched": result.search_attempts[:3]
                })
    
    return stats


def format_path_validation_report(stats: Dict[str, Any]) -> str:
    """Format path validation statistics as human-readable report."""
    lines = [
        "=" * 60,
        "PREDICTION PATH VALIDATION REPORT",
        "=" * 60,
        f"Total rows: {stats['total_rows']}",
        f"Found: {stats['found']} ({100*stats['found']/max(1,stats['total_rows']):.1f}%)",
        f"Not found: {stats['not_found']} ({100*stats['not_found']/max(1,stats['total_rows']):.1f}%)",
        "",
        "Found by strategy:",
    ]
    
    for strategy, count in sorted(stats["found_by_strategy"].items(), key=lambda x: -x[1]):
        lines.append(f"  {strategy}: {count}")
    
    lines.append("")
    lines.append("Found rate by mode:")
    for mode, data in stats["found_by_mode"].items():
        rate = 100 * data["found"] / max(1, data["total"])
        lines.append(f"  {mode}: {data['found']}/{data['total']} ({rate:.1f}%)")
    
    lines.append("")
    lines.append("Found rate by noise:")
    for noise, data in stats["found_by_noise"].items():
        rate = 100 * data["found"] / max(1, data["total"])
        lines.append(f"  {noise}: {data['found']}/{data['total']} ({rate:.1f}%)")
    
    lines.append("")
    lines.append("Found rate by level:")
    for level, data in sorted(stats["found_by_level"].items()):
        rate = 100 * data["found"] / max(1, data["total"])
        lines.append(f"  {level}: {data['found']}/{data['total']} ({rate:.1f}%)")
    
    if stats["missing_paths"]:
        lines.append("")
        lines.append(f"First {len(stats['missing_paths'])} missing paths:")
        for i, mp in enumerate(stats["missing_paths"][:20]):
            lines.append(f"  {i+1}. {mp['sid']} | {mp['mode']}/{mp['noise']}/{mp['level']}")
            if mp.get("searched"):
                lines.append(f"      Tried: {mp['searched'][0]}")
    
    lines.append("=" * 60)
    return "\n".join(lines)
