"""
Re-run the lowest-accuracy dog features with stronger robustness-oriented
hyperparameters.

Usage:
  python run_dog_lowacc_retune.py
  python run_dog_lowacc_retune.py --features dog_low_poly1 dog_fox1
  python run_dog_lowacc_retune.py --dry_run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import torch


THIS_DIR = Path(__file__).resolve().parent

# Lowest id_acc_wm features from detail_afhq_dog.csv.
RETUNE_FEATURES = [
    "dog_low_poly1",
    "dog_fox1",
    "dog_pixel_art1",
    "dog_cat1",
    "dog_lion1",
]

# Stronger watermark / identity settings for harder semantic shifts.
SHARED_ARGS = [
    "--config", "configs/afhq.yml",
    "--num_user", "100",
    "--fp16", "1",
    "--bs_train", "4",
    "--ip2p_steps", "6",
    "--embed_dim", "128",
    "--user_dim", "64",
    "--writer_hidden", "96",
    "--writer_blocks", "4",
    "--wm_strength", "0.09",
    "--carr_lambda", "0.70",
    "--sim_lambda", "1.6",
    "--id_lambda", "4.2",
    "--out_decode_lambda", "1.8",
    "--cons_lambda", "1.0",
    "--n_train_img", "1404",
    "--n_iter", "220",
    "--lr", "1.5e-4",
    "--warmup_no_edit_iters", "45",
    "--edit_ramp_iters", "120",
    "--save_images", "1",
    "--auto_eval", "1",
    "--clean_image_out", "1",
    "--batch_pbar", "0",
    "--seed", "1234",
]


def build_command(args, feature: str) -> List[str]:
    return [
        "torchrun",
        "--master_port", str(args.master_port),
        "--nproc_per_node", str(args.nproc_per_node),
        "main.py",
        *SHARED_ARGS,
        "--instruction", feature,
    ]


def run_summary() -> None:
    cmd = [sys.executable, "summary.py", "--report_dir", str(THIS_DIR / "report")]
    print("[summary_cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=THIS_DIR, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Retune lowest-accuracy dog features")
    n_gpus = torch.cuda.device_count()
    p.add_argument("--nproc_per_node", type=int, default=n_gpus)
    p.add_argument("--master_port", type=int, default=29602)
    p.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Subset of features to rerun. Default: the 5 lowest id_acc_wm dog features.",
    )
    p.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    return p.parse_args()


def main():
    args = parse_args()
    features = args.features if args.features else RETUNE_FEATURES
    env = os.environ.copy()

    print(f"[retune] features: {features}")
    print(f"[retune] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print("[retune] tuned for harder semantic edits: stronger id path, stronger watermark, longer training")
    print()

    for idx, feature in enumerate(features, 1):
        cmd = build_command(args, feature)
        print(f"[{idx}/{len(features)}] {feature}")
        print("[cmd]", " ".join(cmd))

        if args.dry_run:
            continue

        subprocess.run(cmd, cwd=THIS_DIR, env=env, check=True)

    if args.dry_run:
        print("\n[retune] dry run finished")
        return

    print("\n[retune] all done, running summary.py ...")
    run_summary()


if __name__ == "__main__":
    main()
