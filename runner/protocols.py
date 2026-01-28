"""
Protocol case generation for AIO25 NoisySAM benchmark.

Protocols:
  P0: Clean baseline (L0, no noise)
  P1: Coupled levels (p and severity increase together)
  P2a: OFAT - sweep severity with fixed p
  P2b: OFAT - sweep probability with fixed severity
  P3: Grid search over p × severity combinations
  
Includes schedule generator for systematic noise evaluation.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass(frozen=True)
class ProtocolCase:
    """
    Single protocol case specification.
    
    Attributes:
        protocol: Protocol type (P0/P1/P2a/P2b/P3)
        noise_name: Noise type (gaussian, poisson, etc.)
        level: Level name (L0..L4 or custom)
        p: Probability of noise application
        params: Noise-specific severity parameters
        intensity_scalar: Overall intensity in [0,1] for analysis
        noise_seed: Random seed for noise generation
    """
    protocol: str
    noise_name: str
    level: str
    p: float
    params: Dict
    intensity_scalar: float = 0.0
    noise_seed: int = 42


@dataclass
class NoiseSpec:
    """
    Full noise specification for schedule generation.
    """
    noise_type: str
    protocol: str
    level: str
    p: float
    severity_params: Dict
    severity_scalar: float
    intensity_scalar: float
    noise_seed: int = 42


def get_level_intensity_scalar(level: str, cfg: Dict[str, Any], *, strict: bool = True) -> float:
    """
    Get intensity scalar for a level using cfg['levels']['intensity_scalars'] FIRST.

    - Normalizes 'level' (strip, upper).
    - Accepts numeric levels like 0/1/2/3/4 or "0"/"1"... and converts to "L0".."L4".
    - If strict=True: raise KeyError when not found (recommended for benchmark correctness).
      If strict=False: fall back to evenly spaced mapping.
    """
    # 1) Normalize level to canonical form "Lx"
    lv = str(level).strip().upper()

    # Convert "0".."4" to "L0".."L4"
    if lv.isdigit():
        lv = f"L{int(lv)}"

    # Convert "LEVEL_1" or "LEVEL1" etc. (optional safety)
    if lv.startswith("LEVEL"):
        # keep only trailing digits if present
        digits = "".join(ch for ch in lv if ch.isdigit())
        if digits:
            lv = f"L{int(digits)}"

    levels_cfg = cfg.get("levels", {}) or {}
    intensity_scalars = levels_cfg.get("intensity_scalars", {}) or {}

    # 2) Primary: use config mapping exactly
    if lv in intensity_scalars:
        return float(intensity_scalars[lv])

    # 3) If strict, fail fast (best practice)
    if strict:
        raise KeyError(
            f"Unknown level={level!r} (normalized={lv!r}). "
            f"Available in cfg['levels']['intensity_scalars']: {sorted(intensity_scalars.keys())}"
        )

    # 4) Non-strict fallback: evenly spaced based on cfg['levels']['names'] (or default)
    level_names = levels_cfg.get("names", ["L0", "L1", "L2", "L3", "L4"])
    level_names = [str(x).strip().upper() for x in level_names]
    if lv in level_names:
        idx = level_names.index(lv)
        denom = max(1, len(level_names) - 1)
        return idx / denom

    # 5) Last resort: parse Lx
    if lv.startswith("L"):
        try:
            idx = int(lv[1:])
            return idx / 4.0
        except ValueError:
            pass

    return 0.5



def get_schedule(
    noise_type: str,
    protocol: str,
    levels: List[str],
    coupled_presets: Dict,
    p_fixed: float = 0.8,
    severity_ref_level: str = "L3",
    p_values: List[float] = None,
    noise_seed: int = 42,
    cfg: dict = None
) -> List[NoiseSpec]:
    """
    Generate noise schedule for a specific noise type and protocol.
    
    Args:
        noise_type: Noise type name
        protocol: Protocol (P0/P1/P2a/P2b/P3)
        levels: List of level names
        coupled_presets: Coupled presets from config
        p_fixed: Fixed p for P2a
        severity_ref_level: Reference level for P2b
        p_values: P values for P2b and P3
        noise_seed: Random seed
        cfg: Full config for intensity scalars
        
    Returns:
        List of NoiseSpec for the schedule
    """
    cfg = cfg or {}
    schedule = []
    
    if noise_type not in coupled_presets:
        return schedule
    
    lv_map = coupled_presets[noise_type]
    
    if protocol == "P0":
        # Clean baseline
        schedule.append(NoiseSpec(
            noise_type="clean",
            protocol="P0",
            level="L0",
            p=0.0,
            severity_params={},
            severity_scalar=0.0,
            intensity_scalar=0.0,
            noise_seed=noise_seed
        ))
        
    elif protocol == "P1":
        # Coupled: p and severity both increase
        for lv in levels:
            if lv == "L0":
                continue
            if lv not in lv_map:
                continue
            
            params = dict(lv_map[lv])
            p = float(params.pop("p", 1.0))
            intensity_scalar = get_level_intensity_scalar(lv, cfg)
            
            schedule.append(NoiseSpec(
                noise_type=noise_type,
                protocol="P1",
                level=lv,
                p=p,
                severity_params=params,
                severity_scalar=intensity_scalar,
                intensity_scalar=p * intensity_scalar,
                noise_seed=noise_seed
            ))
            
    elif protocol == "P2a":
        # OFAT: sweep severity, p fixed
        for lv in levels:
            if lv == "L0":
                continue
            if lv not in lv_map:
                continue
            
            params = dict(lv_map[lv])
            params.pop("p", None)
            severity_scalar = get_level_intensity_scalar(lv, cfg)
            
            schedule.append(NoiseSpec(
                noise_type=noise_type,
                protocol="P2a",
                level=lv,
                p=p_fixed,
                severity_params=params,
                severity_scalar=severity_scalar,
                intensity_scalar=p_fixed * severity_scalar,
                noise_seed=noise_seed
            ))
            
    elif protocol == "P2b":
        # OFAT: sweep p, severity fixed at reference level
        if severity_ref_level not in lv_map:
            return schedule
        
        ref_params = dict(lv_map[severity_ref_level])
        ref_params.pop("p", None)
        ref_severity = get_level_intensity_scalar(severity_ref_level, cfg)
        
        p_values = p_values or [0.2, 0.5, 0.8, 1.0]
        p_lv_map = {"L1": p_values[0] if len(p_values) > 0 else 0.2,
                    "L2": p_values[1] if len(p_values) > 1 else 0.5,
                    "L3": p_values[2] if len(p_values) > 2 else 0.8,
                    "L4": p_values[3] if len(p_values) > 3 else 1.0}
        
        for lv in levels:
            if lv == "L0" or lv not in p_lv_map:
                continue
            
            p = p_lv_map[lv]
            schedule.append(NoiseSpec(
                noise_type=noise_type,
                protocol="P2b",
                level=lv,
                p=p,
                severity_params=dict(ref_params),
                severity_scalar=ref_severity,
                intensity_scalar=p * ref_severity,
                noise_seed=noise_seed
            ))
            
    elif protocol == "P3":
        # Grid: p × severity
        p_values = p_values or [0.2, 0.5, 0.8]
        severity_levels = [lv for lv in levels if lv != "L0" and lv in lv_map]
        
        for sev_lv in severity_levels:
            params = dict(lv_map[sev_lv])
            params.pop("p", None)
            severity_scalar = get_level_intensity_scalar(sev_lv, cfg)
            
            for p in p_values:
                level_tag = f"{sev_lv}_p{p}"
                schedule.append(NoiseSpec(
                    noise_type=noise_type,
                    protocol="P3",
                    level=level_tag,
                    p=p,
                    severity_params=dict(params),
                    severity_scalar=severity_scalar,
                    intensity_scalar=p * severity_scalar,
                    noise_seed=noise_seed
                ))
    
    return schedule


def build_protocol_cases(cfg: dict) -> List[ProtocolCase]:
    """
    Build all protocol cases from config.
    
    Args:
        cfg: Full config dictionary
        
    Returns:
        List of ProtocolCase for experiment execution
    """
    levels = cfg["levels"]["names"]
    enabled = cfg["protocols"]["enabled"]

    coupled = cfg["protocols"].get("coupled_presets", {})
    ofat = cfg["protocols"].get("ofat", {})
    grid = cfg["protocols"].get("grid", {})
    
    noise_cfg = cfg.get("noise_config", {})
    base_seed = noise_cfg.get("noise_seed", 42)
    n_seeds = noise_cfg.get("n_noise_seeds", 1)
    
    # Debug filters
    debug = cfg.get("debug", {})
    noise_filter = debug.get("noise_types")
    level_filter = debug.get("levels")

    cases: List[ProtocolCase] = []

    # P0: clean baseline (L0, no noise)
    if "P0" in enabled:
        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx
            cases.append(ProtocolCase(
                protocol="P0",
                noise_name="clean",
                level="L0",
                p=0.0,
                params={},
                intensity_scalar=0.0,
                noise_seed=seed
            ))

    # P1: coupled levels
    if "P1" in enabled:
        for noise_name, lv_map in coupled.items():
            if noise_filter and noise_name not in noise_filter:
                continue
            
            for lv in levels:
                if lv == "L0":
                    continue
                if level_filter and lv not in level_filter:
                    continue
                if lv not in lv_map:
                    continue
                
                d = dict(lv_map[lv])
                p = float(d.pop("p", 1.0))
                intensity_scalar = get_level_intensity_scalar(lv, cfg)
                
                for seed_idx in range(n_seeds):
                    seed = base_seed + seed_idx
                    cases.append(ProtocolCase(
                        protocol="P1",
                        noise_name=noise_name,
                        level=lv,
                        p=p,
                        params=d,
                        intensity_scalar=p * intensity_scalar,
                        noise_seed=seed
                    ))

    # P2: OFAT
    if "P2" in enabled:
        p_fixed = float(ofat.get("p_fixed", 1.0))
        severity_ref_level = str(ofat.get("severity_ref_level", "L3"))
        
        for noise_name, lv_map in coupled.items():
            if noise_filter and noise_name not in noise_filter:
                continue
            
            # P2a: sweep severity, p fixed
            for lv in levels:
                if lv == "L0":
                    continue
                if level_filter and lv not in level_filter:
                    continue
                if lv not in lv_map:
                    continue
                
                d = dict(lv_map[lv])
                d.pop("p", None)
                intensity_scalar = get_level_intensity_scalar(lv, cfg)
                
                for seed_idx in range(n_seeds):
                    seed = base_seed + seed_idx
                    cases.append(ProtocolCase(
                        protocol="P2a",
                        noise_name=noise_name,
                        level=lv,
                        p=p_fixed,
                        params=d,
                        intensity_scalar=p_fixed * intensity_scalar,
                        noise_seed=seed
                    ))

            # P2b: sweep p, severity fixed at ref level
            if severity_ref_level in lv_map:
                ref = dict(lv_map[severity_ref_level])
                ref.pop("p", None)
                ref_intensity = get_level_intensity_scalar(severity_ref_level, cfg)
                
                p_lv = {"L1": 0.2, "L2": 0.5, "L3": 0.8, "L4": 1.0}
                
                for lv, p in p_lv.items():
                    if level_filter and lv not in level_filter:
                        continue
                    
                    for seed_idx in range(n_seeds):
                        seed = base_seed + seed_idx
                        cases.append(ProtocolCase(
                            protocol="P2b",
                            noise_name=noise_name,
                            level=lv,
                            p=float(p),
                            params=dict(ref),
                            intensity_scalar=float(p) * ref_intensity,
                            noise_seed=seed
                        ))

    # P3: grid p x severity
    if "P3" in enabled and grid.get("enabled", False):
        p_values = list(grid.get("p_values", [0.2, 0.5, 0.8]))
        sev_levels = list(grid.get("severity_levels", ["L1", "L2", "L3"]))
        
        for noise_name, lv_map in coupled.items():
            if noise_filter and noise_name not in noise_filter:
                continue
            
            for sev in sev_levels:
                if sev not in lv_map:
                    continue
                if level_filter and sev not in level_filter:
                    continue
                
                d = dict(lv_map[sev])
                d.pop("p", None)
                sev_intensity = get_level_intensity_scalar(sev, cfg)
                
                for p in p_values:
                    level_tag = f"{sev}_p{p}"
                    
                    for seed_idx in range(n_seeds):
                        seed = base_seed + seed_idx
                        cases.append(ProtocolCase(
                            protocol="P3",
                            noise_name=noise_name,
                            level=level_tag,
                            p=float(p),
                            params=dict(d),
                            intensity_scalar=float(p) * sev_intensity,
                            noise_seed=seed
                        ))

    return cases


def filter_cases_by_debug(cases: List[ProtocolCase], cfg: dict) -> List[ProtocolCase]:
    """
    Filter protocol cases based on debug settings.
    
    Args:
        cases: List of all protocol cases
        cfg: Config with debug settings
        
    Returns:
        Filtered list of cases
    """
    debug = cfg.get("debug", {})
    
    noise_filter = debug.get("noise_types")
    level_filter = debug.get("levels")
    
    filtered = []
    for case in cases:
        if noise_filter and case.noise_name not in noise_filter and case.noise_name != "clean":
            continue
        if level_filter and case.level not in level_filter and not case.level.startswith(tuple(level_filter or [])):
            continue
        filtered.append(case)
    
    return filtered
