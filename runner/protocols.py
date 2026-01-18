from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ProtocolCase:
    protocol: str          # P0/P1/P2a/P2b/P3
    noise_name: str        # gaussian/...
    level: str             # L0..L4
    p: float               # probability
    params: Dict           # severity params


def build_protocol_cases(cfg: dict) -> List[ProtocolCase]:
    levels = cfg["levels"]["names"]
    enabled = cfg["protocols"]["enabled"]

    coupled = cfg["protocols"].get("coupled_presets", {})
    ofat = cfg["protocols"].get("ofat", {})
    grid = cfg["protocols"].get("grid", {})

    cases: List[ProtocolCase] = []

    # P0: clean baseline (L0, no noise)
    if "P0" in enabled:
        cases.append(ProtocolCase(protocol="P0", noise_name="clean", level="L0", p=0.0, params={}))

    # P1: coupled levels
    if "P1" in enabled:
        for noise_name, lv_map in coupled.items():
            for lv in levels:
                if lv == "L0":
                    continue
                if lv not in lv_map:
                    continue
                d = dict(lv_map[lv])
                p = float(d.pop("p", 1.0))
                cases.append(ProtocolCase(protocol="P1", noise_name=noise_name, level=lv, p=p, params=d))

    # P2: OFAT
    if "P2" in enabled:
        p_fixed = float(ofat.get("p_fixed", 1.0))
        severity_ref_level = str(ofat.get("severity_ref_level", "L3"))
        for noise_name, lv_map in coupled.items():
            # P2a: sweep severity, p fixed
            for lv in levels:
                if lv == "L0":
                    continue
                if lv not in lv_map:
                    continue
                d = dict(lv_map[lv])
                d.pop("p", None)
                cases.append(ProtocolCase(protocol="P2a", noise_name=noise_name, level=lv, p=p_fixed, params=d))

            # P2b: sweep p, severity fixed at ref level
            if severity_ref_level in lv_map:
                ref = dict(lv_map[severity_ref_level])
                ref.pop("p", None)
                # p increases across L1..L4
                p_lv = { "L1": 0.2, "L2": 0.5, "L3": 0.8, "L4": 1.0 }
                for lv, p in p_lv.items():
                    cases.append(ProtocolCase(protocol="P2b", noise_name=noise_name, level=lv, p=float(p), params=dict(ref)))

    # P3: grid p x severity
    if "P3" in enabled and grid.get("enabled", False):
        p_values = list(grid.get("p_values", [0.2, 0.5, 0.8]))
        sev_levels = list(grid.get("severity_levels", ["L1", "L2", "L3"]))
        for noise_name, lv_map in coupled.items():
            for sev in sev_levels:
                if sev not in lv_map:
                    continue
                d = dict(lv_map[sev])
                d.pop("p", None)
                for p in p_values:
                    # encode grid as its own “level tag”
                    cases.append(ProtocolCase(protocol="P3", noise_name=noise_name, level=f"{sev}_p{p}", p=float(p), params=dict(d)))

    return cases
