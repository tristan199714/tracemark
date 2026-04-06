"""
Re-run strong-stylization features (pixar, modigliani, nicolas, obama, jocker,
zombie, gogh, elf) with the same hyperparameters as run_human1_batch.py.

Usage:
  python run_stylized_retune.py                          # run all
  python run_stylized_retune.py --features pixar1 modigliani1  # run specific
  python run_stylized_retune.py --dry_run                # preview commands
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import torch

THIS_DIR = Path(__file__).resolve().parent

# ── features to retune ──────────────────────────────────────────────
STYLIZED_FEATURES = [
    "pixar1",
    "modigliani1",
    "nicolas1",
    "obama1",
    "jocker1",
    "zombie1",
    "gogh1",
    "elf1",
]

# ── tuned args (diff from default marked with comments) ─────────────
SHARED_ARGS = [
    "--config",           "configs/celeba.yml",
    "--num_user",         "100",
    "--fp16",             "1",
    "--bs_train",         "8",
    "--ip2p_steps",       "6",
    "--embed_dim",        "128",
    "--user_dim",         "64",
    "--writer_hidden",    "96",
    "--writer_blocks",    "4",
    "--wm_strength",      "0.07",
    "--carr_lambda",      "0.85",
    "--sim_lambda",       "2.0",
    "--id_lambda",        "3.2",
    "--out_decode_lambda","1.4",
    "--cons_lambda",      "0.8",
    "--n_train_img",      "1404",
    "--n_iter",           "170",
    "--lr",               "1.8e-4",
    "--warmup_no_edit_iters", "35",
    "--edit_ramp_iters",  "90",
    "--save_images",      "1",
    "--auto_eval",        "1",
    "--clean_image_out",  "1",
    "--batch_pbar",       "0",
    "--seed",             "1234",
]


OUT_ROOT = THIS_DIR / "out"


def has_existing_result(feature: str) -> bool:
    feature_dir = OUT_ROOT / feature
    if (feature_dir / "summary.json").exists() or (feature_dir / "checkpoint.pt").exists():
        return True
    if feature_dir.is_dir():
        for p in feature_dir.iterdir():
            if p.is_dir() and ((p / "summary.json").exists() or (p / "checkpoint.pt").exists()):
                return True
    return False


def build_command(args, feature: str) -> List[str]:
    return [
        "torchrun",
        "--master_port", str(args.master_port),
        "--nproc_per_node", str(args.nproc_per_node),
        "main.py",
        *SHARED_ARGS,
        "--instruction", feature,
    ]


def parse_args():
    p = argparse.ArgumentParser(description="Retune strong-stylization features")
    n_gpus = torch.cuda.device_count()
    p.add_argument("--nproc_per_node", type=int, default=n_gpus)
    p.add_argument("--master_port", type=int, default=29601)
    p.add_argument("--features", nargs="+", default=None,
                   help="Subset of features to run (default: all 8)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running")
    return p.parse_args()


def main():
    args = parse_args()
    features = args.features if args.features else STYLIZED_FEATURES

    env = os.environ.copy()

    print(f"[retune] features: {features}")
    print(f"[retune] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"[retune] using same params as run_human1_batch.py")
    print()

    for idx, feature in enumerate(features, 1):
        if has_existing_result(feature):
            print(f"[{idx}/{len(features)}] {feature} [skip] already has result")
            continue

        cmd = build_command(args, feature)
        print(f"[{idx}/{len(features)}] {feature}")
        print("[cmd]", " ".join(cmd))

        if args.dry_run:
            continue

        subprocess.run(cmd, cwd=THIS_DIR, env=env, check=True)

    if args.dry_run:
        print("\n[retune] dry run finished")
        return

    # run summary
    print("\n[retune] all done, running summary.py ...")
    subprocess.run(
        [sys.executable, "summary.py", "--report_dir", str(THIS_DIR / "report")],
        cwd=THIS_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
