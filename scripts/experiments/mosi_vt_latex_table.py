#!/usr/bin/env python3
"""Print MOSI VT results as LaTeX table rows (CEU-S & Test Acc)."""
import json
import math
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "reports"

DISPLAY = {
    "MCR.json":                   "MCR",
    "MCR_NoiseInput.json":        "MCR NoiseInput",
    "MCR_NoiseLatent.json":       "MCR NoiseLatent",
    "MCR_ZeroInput.json":         "MCR ZeroInput",
    "MCR_ZeroLatent.json":        "MCR ZeroLatent",
    "MLB.json":                   "MLB",
    "MMPareto.json":              "MMPareto",
    "OGM.json":                   "OGM",
    "ens.json":                   "Ensemble",
    "joint_training.json":        "Joint Training",
    "multiloss.json":             "Multi-Loss",
    "pre_finetuned.json":         "Pre Fine-tuned",
    "pre_frozen.json":            "Pre Frozen",
    "synprom_RMask_nopre.json":   "SynProm (no-pre)",
    "synprom_RMask.json":         "SynProm RMask",
    "synprom_RMask_learned.json": "SynProm Learned",
    "synprom_RMask_random.json":  "SynProm Random",
}


def latex_val(mean, std):
    if math.isnan(mean):
        return "--"
    return f"{mean:.1f}{{\\tiny$\\pm${std:.1f}}}"


def latest_table():
    files = sorted(REPORTS_DIR.glob("mosi_vt_method_table_*.json"))
    if not files:
        raise FileNotFoundError(f"No mosi_vt_method_table_*.json in {REPORTS_DIR}")
    return files[-1]


def main():
    path = latest_table()
    print(f"% Source: {path.name}")
    data = json.loads(path.read_text())

    print(r"% Method  &  CEU-S  &  Test Acc  \\")
    for r in data.get("methods", []):
        name = DISPLAY.get(r["method"], r["method"])
        if r.get("status") != "ok":
            print(f"{name:<22} & -- & -- \\\\")
            continue
        ceu = latex_val(r["test_ceu_synergy_mean_pct"], r["test_ceu_synergy_std_pct"])
        acc = latex_val(r["test_acc_mean_pct"], r["test_acc_std_pct"])
        print(f"{name:<22} & {ceu} & {acc} \\\\")


if __name__ == "__main__":
    main()
