#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path
from typing import Set


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = REPO_ROOT / "models" / "Synergy_Models_Dec.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ESNLI" / "default_config_esnli_smoke.json"
FULL_CONFIGS = [
    "full_lora.json",
    "full_text_lora.json",
    "full_image_lora.json",
    "full_image_frozen.json",
    "full_synib.json",
    "full_mcr.json",
    "full_mmpareto.json",
    "full_dnr.json",
    "full_reconboost.json",
]


def load_declared_classes(path: Path) -> Set[str]:
    tree = ast.parse(path.read_text())
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def load_model_class(cfg_path: Path) -> str:
    payload = json.loads(cfg_path.read_text())
    return payload["model"]["model_class"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check ESNLI Qwen config/model wiring.")
    parser.add_argument(
        "--configs-dir",
        default=str(REPO_ROOT / "configs" / "ESNLI"),
        help="Directory containing ESNLI configs.",
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Also validate smoke_*.json configs.",
    )
    args = parser.parse_args()

    cfg_dir = Path(args.configs_dir)
    class_names = load_declared_classes(MODEL_FILE)

    targets = [cfg_dir / name for name in FULL_CONFIGS]
    if args.include_smoke:
        targets.extend(sorted(cfg_dir.glob("smoke_*.json")))

    problems = []
    print(f"[smoke] default config: {DEFAULT_CONFIG}")
    if not DEFAULT_CONFIG.exists():
        problems.append(f"missing default config: {DEFAULT_CONFIG}")

    for cfg in targets:
        if not cfg.exists():
            problems.append(f"missing config: {cfg}")
            continue
        try:
            model_class = load_model_class(cfg)
        except Exception as exc:
            problems.append(f"{cfg}: failed to parse JSON/model_class: {exc}")
            continue

        if model_class not in class_names:
            problems.append(f"{cfg}: model_class '{model_class}' not declared in {MODEL_FILE.name}")
            continue

        print(f"[ok] {cfg.relative_to(REPO_ROOT)} -> {model_class}")

    if problems:
        print("[smoke] FAIL")
        for item in problems:
            print(f"  - {item}")
        return 1

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
