#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import torch  # noqa: E402
except Exception:
    torch = None


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
VAL_ACC_RE = re.compile(r"\bAcc_combined:\s*([0-9]+(?:\.[0-9]+)?)\b")
TEST_ACC_RE = re.compile(r"\bTest_Acc_combined:\s*([0-9]+(?:\.[0-9]+)?)\b")
TEST_CEU_RE = re.compile(r"\bT_CEU_S:\s*([0-9]+(?:\.[0-9]+)?)\b")
MISSING_RE = re.compile(r"We could not load ")


def _split_csv(x: str) -> List[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if hasattr(x, "item"):
            x = x.item()
        return float(x)
    except Exception:
        return None


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.pstdev(vals))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _load_ckpt(path: str) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to inspect checkpoints.")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_save_base_and_template(config_path: str, default_config: str) -> Tuple[str, str]:
    merged = _merge_dicts(_load_json(default_config), _load_json(config_path))
    model = merged.get("model", {})
    save_base = model.get("save_base_dir", "")
    save_tpl = model["save_dir"]
    return save_base, save_tpl


def _suffix_for_lrwd(fold: int, lr: str, wd: str, validate_with: str = "accuracy") -> str:
    return f"fold{fold}_vld{validate_with}_lr{lr}_wd{wd}"


def _checkpoint_path_for(config_path: str, default_config: str, fold: int, lr: str, wd: str, validate_with: str) -> str:
    save_base, save_tpl = _resolve_save_base_and_template(config_path, default_config)
    save_name = save_tpl.format(_suffix_for_lrwd(fold, lr, wd, validate_with=validate_with))
    return os.path.join(save_base, save_name) if save_base else save_name


def _read_acc_metrics(ckpt_path: str, validate_with: str = "accuracy") -> Dict[str, float]:
    ckpt = _load_ckpt(ckpt_path)
    out: Dict[str, float] = {}
    try:
        out["val_acc"] = float(_to_float(ckpt["logs"]["best_logs"][f"best_v{validate_with}"]["acc"]["combined"]))
    except Exception:
        out["val_acc"] = math.nan
    try:
        out["test_acc"] = float(_to_float(ckpt["post_test_results"]["acc"]["combined"]))
    except Exception:
        out["test_acc"] = math.nan
    return out


def _select_best_lrwd(config_path: str, default_config: str, folds: List[int], lrs: List[str], wds: List[str], validate_with: str) -> Optional[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lr in lrs:
        for wd in wds:
            val_accs: List[float] = []
            test_accs: List[float] = []
            ok = True
            for fold in folds:
                ckpt_path = _checkpoint_path_for(config_path, default_config, fold, lr, wd, validate_with)
                if not os.path.exists(ckpt_path):
                    ok = False
                    break
                metrics = _read_acc_metrics(ckpt_path, validate_with=validate_with)
                if math.isnan(metrics["val_acc"]):
                    ok = False
                    break
                val_accs.append(metrics["val_acc"])
                if not math.isnan(metrics["test_acc"]):
                    test_accs.append(metrics["test_acc"])
            if not ok:
                continue
            val_mean, val_std = _mean_std(val_accs)
            test_mean, test_std = _mean_std(test_accs)
            rows.append(
                {
                    "lr": lr,
                    "wd": wd,
                    "val_acc_fold": val_accs,
                    "test_acc_fold": test_accs,
                    "val_acc_mean": val_mean,
                    "val_acc_std": val_std,
                    "test_acc_mean": test_mean,
                    "test_acc_std": test_std,
                }
            )
    if not rows:
        return None
    rows.sort(key=lambda r: (r["val_acc_mean"], -r["val_acc_std"], r["test_acc_mean"]), reverse=True)
    best = dict(rows[0])
    best["top3"] = rows[:3]
    return best


def _run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None, dry_run: bool = False) -> int:
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, check=False)
    return int(proc.returncode)


def _run_cmd_capture(cmd: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return int(proc.returncode), proc.stdout


def _parse_show_output(text: str) -> Optional[Dict[str, Any]]:
    clean = ANSI_RE.sub("", text)
    if MISSING_RE.search(clean):
        return None
    val_accs = [float(m.group(1)) for m in VAL_ACC_RE.finditer(clean)]
    test_accs = [float(m.group(1)) for m in TEST_ACC_RE.finditer(clean)]
    test_ceus = [float(m.group(1)) for m in TEST_CEU_RE.finditer(clean)]
    if not val_accs:
        return {"error": "Could not parse validation accuracy", "raw_output": clean}
    val_mean, val_std = _mean_std(val_accs)
    test_mean, test_std = _mean_std(test_accs) if test_accs else (math.nan, math.nan)
    ceu_mean, ceu_std = _mean_std(test_ceus) if test_ceus else (math.nan, math.nan)
    return {
        "val_acc_fold_pct": val_accs,
        "test_acc_fold_pct": test_accs,
        "test_ceu_synergy_fold_raw": test_ceus,
        "val_acc_mean_pct": val_mean,
        "val_acc_std_pct": val_std,
        "test_acc_mean_pct": test_mean,
        "test_acc_std_pct": test_std,
        "test_ceu_synergy_mean_raw": ceu_mean,
        "test_ceu_synergy_std_raw": ceu_std,
        "test_ceu_synergy_mean_pct": 100.0 * ceu_mean if not math.isnan(ceu_mean) else math.nan,
        "test_ceu_synergy_std_pct": 100.0 * ceu_std if not math.isnan(ceu_std) else math.nan,
        "raw_output": clean,
    }


def _show_method(config_path: str, default_config: str, lr: str, wd: str, validate_with: str, python_bin: str, gpu: Optional[str]) -> Tuple[int, Optional[Dict[str, Any]], str, List[str]]:
    cmd = [
        python_bin,
        "scripts/entrypoints/show.py",
        "--config",
        config_path,
        "--default_config",
        default_config,
        "--fold",
        "0",
        "--lr",
        lr,
        "--wd",
        wd,
        "--validate_with",
        validate_with,
    ]
    env = os.environ.copy()
    if gpu is not None and gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    rc, out = _run_cmd_capture(cmd, env=env)
    return rc, _parse_show_output(out), out, cmd


def _safe_rel(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(_REPO_ROOT).resolve()))
    except Exception:
        return path


def _patch_encoder_paths(config_path: str, video_tpl: str, text_tpl: str, dry_run: bool = False) -> bool:
    data = _load_json(config_path)
    changed = False
    encs = data.get("model", {}).get("encoders", [])
    for enc in encs:
        pre = enc.get("pretrainedEncoder")
        if not isinstance(pre, dict):
            continue
        args = enc.get("args", {})
        modality = args.get("modality")
        if modality == "c":
            if pre.get("dir") != video_tpl:
                pre["dir"] = video_tpl
                changed = True
        elif modality == "g":
            if pre.get("dir") != text_tpl:
                pre["dir"] = text_tpl
                changed = True
    if changed and not dry_run:
        _dump_json(config_path, data)
    return changed


def _patch_ceu_paths(config_path: str, ceu_val: str, ceu_test: str, dry_run: bool = False) -> bool:
    data = _load_json(config_path)
    model = data.setdefault("model", {})
    ceu = model.get("ceu")
    changed = False
    if not isinstance(ceu, dict):
        model["ceu"] = {"val": ceu_val, "test": ceu_test}
        changed = True
    else:
        if ceu.get("val") != ceu_val:
            ceu["val"] = ceu_val
            changed = True
        if ceu.get("test") != ceu_test:
            ceu["test"] = ceu_test
            changed = True
    if changed and not dry_run:
        _dump_json(config_path, data)
    return changed


def _discover_methods(release_dir: str, syn_dir: Optional[str]) -> List[str]:
    methods: List[str] = []
    if os.path.isdir(release_dir):
        for p in sorted(Path(release_dir).glob("*.json")):
            if p.name.startswith("unimodal_"):
                continue
            methods.append(str(p))
    if syn_dir and os.path.isdir(syn_dir):
        for p in sorted(Path(syn_dir).glob("*.json")):
            methods.append(str(p))
    return methods


def _discover_unimodals(release_dir: str) -> List[str]:
    out = []
    for name in ["unimodal_video.json", "unimodal_text.json", "unimodal_audio.json"]:
        p = os.path.join(release_dir, name)
        if os.path.exists(p):
            out.append(p)
    return out


def _state_path(args: argparse.Namespace) -> str:
    return os.path.join(args.out_dir, f"{args.dataset}_vt_workflow_state.json")


def _load_state(args: argparse.Namespace) -> Dict[str, Any]:
    path = _state_path(args)
    if os.path.exists(path):
        return _load_json(path)
    return {
        "dataset": args.dataset,
        "default_config": args.default_config,
        "release_dir": args.release_dir,
        "syn_dir": args.syn_dir,
        "validate_with": args.validate_with,
        "folds": args.folds,
        "unimodals": {},
        "best_rmask_nopre": None,
    }


def _save_state(args: argparse.Namespace, state: Dict[str, Any]) -> None:
    _dump_json(_state_path(args), state)


def _train_grid(args: argparse.Namespace, cfgs: List[str], lrs: List[str], wds: List[str]) -> int:
    python_bin = args.python_bin
    env = os.environ.copy()
    if args.gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    rc_any = 0
    for fold in args.folds:
        for cfg in cfgs:
            for lr in lrs:
                for wd in wds:
                    cmd = [
                        python_bin,
                        "scripts/entrypoints/train.py",
                        "--config",
                        cfg,
                        "--default_config",
                        args.default_config,
                        "--fold",
                        str(fold),
                        "--lr",
                        lr,
                        "--wd",
                        wd,
                        "--validate_with",
                        args.validate_with,
                    ]
                    rc = _run_cmd(cmd, env=env, dry_run=args.dry_run)
                    if rc != 0:
                        rc_any = rc
    return rc_any


def do_train_unimodals(args: argparse.Namespace, state: Dict[str, Any]) -> int:
    return _train_grid(args, _discover_unimodals(args.release_dir), args.unimodal_lrs, args.unimodal_wds)


def do_select_unimodals(args: argparse.Namespace, state: Dict[str, Any]) -> int:
    unimodals = _discover_unimodals(args.release_dir)
    if not unimodals:
        print("No unimodal configs found.", file=sys.stderr)
        return 1
    if args.dry_run:
        for cfg in unimodals:
            print(f"[dry-run] select best lr/wd for {_safe_rel(cfg)} over {len(args.unimodal_lrs) * len(args.unimodal_wds)} combinations")
        return 0
    for cfg in unimodals:
        best = _select_best_lrwd(cfg, args.default_config, args.folds, args.unimodal_lrs, args.unimodal_wds, args.validate_with)
        key = Path(cfg).stem
        state["unimodals"][key] = {
            "config": cfg,
            "best": best,
        }
        if best is None:
            print(f"[{key}] no valid checkpoints found")
        else:
            print(
                f"[{key}] best lr={best['lr']} wd={best['wd']} "
                f"val={best['val_acc_mean']*100:.2f}±{best['val_acc_std']*100:.2f} "
                f"test={best['test_acc_mean']*100:.2f}±{best['test_acc_std']*100:.2f}"
            )
    _save_state(args, state)
    print(f"Saved state: {_state_path(args)}")
    return 0


def do_ceu(args: argparse.Namespace, state: Dict[str, Any]) -> int:
    uni_video = state["unimodals"].get("unimodal_video", {}).get("best")
    uni_text = state["unimodals"].get("unimodal_text", {}).get("best")
    if not uni_video or not uni_text:
        print("Need selected unimodal_video and unimodal_text in state first.", file=sys.stderr)
        return 1
    video_cfg = state["unimodals"]["unimodal_video"]["config"]
    text_cfg = state["unimodals"]["unimodal_text"]["config"]
    cmd = [
        args.python_bin,
        "scripts/entrypoints/get_ceu_cli.py",
        "--dataset",
        args.dataset,
        "--default_config",
        args.default_config,
        "--unimodal_configs",
        video_cfg,
        text_cfg,
        "--folds",
        *[str(f) for f in args.folds],
        "--validate_with",
        args.validate_with,
        "--unimodal_lrs",
        uni_video["lr"],
        uni_text["lr"],
        "--unimodal_wds",
        uni_video["wd"],
        uni_text["wd"],
        "--lr",
        uni_video["lr"],
        "--wd",
        uni_video["wd"],
    ]
    env = os.environ.copy()
    if args.gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return _run_cmd(cmd, env=env, dry_run=args.dry_run)


def _rmask_nopre_cfg(args: argparse.Namespace) -> Optional[str]:
    if args.rmask_nopre_config:
        return args.rmask_nopre_config if os.path.exists(args.rmask_nopre_config) else None
    candidates = []
    if args.syn_dir:
        candidates.append(os.path.join(args.syn_dir, "synprom_RMask_nopre.json"))
    candidates.append(os.path.join(args.release_dir, "synprom_RMask_nopre.json"))
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def do_train_rmask_nopre(args: argparse.Namespace, state: Dict[str, Any]) -> int:
    cfg = _rmask_nopre_cfg(args)
    if not cfg:
        print("RMask_nopre config not found.", file=sys.stderr)
        return 1
    return _train_grid(args, [cfg], args.unimodal_lrs, args.unimodal_wds)


def do_select_rmask_nopre(args: argparse.Namespace, state: Dict[str, Any]) -> int:
    cfg = _rmask_nopre_cfg(args)
    if not cfg:
        print("RMask_nopre config not found.", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"[dry-run] select best lr/wd for {_safe_rel(cfg)} over {len(args.unimodal_lrs) * len(args.unimodal_wds)} combinations")
        return 0
    best = _select_best_lrwd(cfg, args.default_config, args.folds, args.unimodal_lrs, args.unimodal_wds, args.validate_with)
    if best is None:
        print("No valid RMask_nopre checkpoints found.", file=sys.stderr)
        return 1
    state["best_rmask_nopre"] = {"config": cfg, **best}
    _save_state(args, state)
    print(
        f"[RMask_nopre] best lr={best['lr']} wd={best['wd']} "
        f"val={best['val_acc_mean']*100:.2f}±{best['val_acc_std']*100:.2f} "
        f"test={best['test_acc_mean']*100:.2f}±{best['test_acc_std']*100:.2f}"
    )
    return 0


def _best_fixed_lrwd(state: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    r = state.get("best_rmask_nopre")
    if r and r.get("lr") and r.get("wd"):
        return str(r["lr"]), str(r["wd"])
    return None


def do_patch_configs(args: argparse.Namespace, state: Dict[str, Any]) -> int:
    vbest = state["unimodals"].get("unimodal_video", {}).get("best")
    tbest = state["unimodals"].get("unimodal_text", {}).get("best")
    if not vbest or not tbest:
        print("Need selected unimodal_video and unimodal_text in state first.", file=sys.stderr)
        return 1
    video_tpl = f"unimodal_video_fold{{}}_vld{args.validate_with}_lr{vbest['lr']}_wd{vbest['wd']}.pth.tar"
    text_tpl = f"unimodal_text_fold{{}}_vld{args.validate_with}_lr{tbest['lr']}_wd{tbest['wd']}.pth.tar"

    targets: List[str] = []
    for p in Path(args.release_dir).glob("*.json"):
        if p.name.startswith("unimodal_"):
            continue
        targets.append(str(p))
    if args.syn_dir and os.path.isdir(args.syn_dir):
        for p in Path(args.syn_dir).glob("*.json"):
            if p.name.startswith("unimodal_"):
                continue
            targets.append(str(p))
    targets = sorted(set(targets))
    if not targets:
        print("No method configs found to patch.")
        return 0

    ceu_dir = os.path.join("./artifacts/ceus", args.dataset)
    ceu_val = os.path.join(ceu_dir, f"{args.dataset}_ceu_val.pkl")
    ceu_test = os.path.join(ceu_dir, f"{args.dataset}_ceu_test.pkl")

    changed_any = False
    for p in targets:
        changed_enc = _patch_encoder_paths(p, video_tpl=video_tpl, text_tpl=text_tpl, dry_run=args.dry_run)
        changed_ceu = _patch_ceu_paths(p, ceu_val=ceu_val, ceu_test=ceu_test, dry_run=args.dry_run)
        tag = "UPDATED" if (changed_enc or changed_ceu) else "ok"
        print(f"[patch] {_safe_rel(p)} {tag}")
        changed_any = changed_any or changed_enc or changed_ceu
    if args.dry_run:
        print(f"[patch] dry-run video={video_tpl} text={text_tpl}")
        print(f"[patch] dry-run ceu_val={ceu_val}")
        print(f"[patch] dry-run ceu_test={ceu_test}")
    else:
        print(f"[patch] video={video_tpl}")
        print(f"[patch] text={text_tpl}")
        print(f"[patch] ceu_val={ceu_val}")
        print(f"[patch] ceu_test={ceu_test}")
    return 0 if changed_any or targets else 1


def do_train_methods(args: argparse.Namespace, state: Dict[str, Any]) -> int:
    lrwd = _best_fixed_lrwd(state)
    if not lrwd:
        print("Need best_rmask_nopre selected first.", file=sys.stderr)
        return 1
    lr, wd = lrwd
    methods = _discover_methods(args.release_dir, args.syn_dir)
    if args.exclude_methods:
        excluded = {m.strip() for m in args.exclude_methods.split(",") if m.strip()}
        methods = [m for m in methods if Path(m).name not in excluded and Path(m).stem not in excluded]
    return _train_grid(args, methods, [lr], [wd])


def do_report(args: argparse.Namespace, state: Dict[str, Any]) -> int:
    lrwd = _best_fixed_lrwd(state)
    if not lrwd:
        print("Need best_rmask_nopre selected first.", file=sys.stderr)
        return 1
    lr, wd = lrwd
    methods = _discover_methods(args.release_dir, args.syn_dir)
    if args.exclude_methods:
        excluded = {m.strip() for m in args.exclude_methods.split(",") if m.strip()}
        methods = [m for m in methods if Path(m).name not in excluded and Path(m).stem not in excluded]

    rows: List[Dict[str, Any]] = []
    for cfg in methods:
        rc, parsed, raw, cmd = _show_method(cfg, args.default_config, lr, wd, args.validate_with, args.python_bin, args.gpu)
        name = _safe_rel(cfg)
        if parsed is None:
            print(f"[report] missing checkpoint: {name}")
            rows.append({"method": name, "status": "missing", "show_cmd": " ".join(cmd)})
            continue
        if isinstance(parsed, dict) and "error" in parsed:
            print(f"[report] parse error: {name}")
            rows.append({"method": name, "status": "parse_error", "error": parsed["error"], "show_cmd": " ".join(cmd)})
            continue
        row = {
            "method": name,
            "status": "ok",
            "fixed_lr": lr,
            "fixed_wd": wd,
            "show_cmd": " ".join(cmd),
            **parsed,
        }
        rows.append(row)
        ceu_str = "NA"
        if not math.isnan(row["test_ceu_synergy_mean_pct"]):
            ceu_str = f"{row['test_ceu_synergy_mean_pct']:.2f}±{row['test_ceu_synergy_std_pct']:.2f}"
        print(
            f"{name}: "
            f"val={row['val_acc_mean_pct']:.2f}±{row['val_acc_std_pct']:.2f} "
            f"test={row['test_acc_mean_pct']:.2f}±{row['test_acc_std_pct']:.2f} "
            f"ceu_s={ceu_str}"
        )
        if args.verbose_report:
            print(raw)

    out_json = os.path.join(args.out_dir, f"{args.dataset}_vt_best_val_vs_test_{Path(_state_path(args)).stem}.json")
    _dump_json(out_json, {"dataset": args.dataset, "fixed_lr": lr, "fixed_wd": wd, "rows": rows})
    print(f"Saved report: {out_json}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="FactorCL VT workflow helper for MOSI/MOSEI.")
    p.add_argument("--dataset", required=True, help="Dataset slug, e.g. mosi or mosei")
    p.add_argument("--default_config", required=True)
    p.add_argument("--release_dir", required=True)
    p.add_argument("--syn_dir", default="")
    p.add_argument("--mode", default="all", choices=[
        "all",
        "train_unimodals",
        "select_unimodals",
        "ceu",
        "train_rmask_nopre",
        "select_rmask_nopre",
        "patch",
        "train_methods",
        "report",
    ])
    p.add_argument("--gpu", default="0")
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--validate_with", default="accuracy")
    p.add_argument("--unimodal_lrs", default="0.001,0.0005,0.0001,0.00005")
    p.add_argument("--unimodal_wds", default="0.001,0.0001,0.00001")
    p.add_argument("--rmask_nopre_config", default="")
    p.add_argument("--exclude_methods", default="", help="Comma-separated filenames/stems to skip from train/report.")
    p.add_argument("--out_dir", default="./artifacts/reports")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--verbose_report", action="store_true")
    args = p.parse_args()

    args.syn_dir = args.syn_dir or ""
    args.unimodal_lrs = _split_csv(args.unimodal_lrs)
    args.unimodal_wds = _split_csv(args.unimodal_wds)
    os.makedirs(args.out_dir, exist_ok=True)

    state = _load_state(args)

    if args.mode == "train_unimodals":
        return do_train_unimodals(args, state)
    if args.mode == "select_unimodals":
        return do_select_unimodals(args, state)
    if args.mode == "ceu":
        return do_ceu(args, state)
    if args.mode == "train_rmask_nopre":
        return do_train_rmask_nopre(args, state)
    if args.mode == "select_rmask_nopre":
        return do_select_rmask_nopre(args, state)
    if args.mode == "patch":
        return do_patch_configs(args, state)
    if args.mode == "train_methods":
        return do_train_methods(args, state)
    if args.mode == "report":
        return do_report(args, state)

    # all
    steps = [
        ("train_unimodals", do_train_unimodals),
        ("select_unimodals", do_select_unimodals),
        ("ceu", do_ceu),
        ("train_rmask_nopre", do_train_rmask_nopre),
        ("select_rmask_nopre", do_select_rmask_nopre),
        ("patch", do_patch_configs),
        ("train_methods", do_train_methods),
        ("report", do_report),
    ]
    for name, fn in steps:
        print(f"\n=== {name} ===")
        rc = fn(args, state)
        if rc != 0:
            return rc
        state = _load_state(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
