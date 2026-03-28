import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
SHARED_ROOT = THIS_DIR.parent / "instructPix2Pix"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from utils.text_dic import SRC_TRG_TXT_DIC

OUT_ROOT = THIS_DIR / "out"

SHARED_ARGS = [
    "--config",           "../instructPix2Pix/configs/church.yml",
    "--num_user",         "100",
    "--fp16",             "1",
    "--bs_train",         "2",
    "--turbo_steps",      "1",
    "--turbo_strength",   "0.55",
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


def get_church_features() -> List[str]:
    return [k for k in SRC_TRG_TXT_DIC if k.endswith("1") and k.startswith("church_")]


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


def run_summary() -> None:
    cmd = [sys.executable, "summary.py", "--report_dir", str(THIS_DIR / "report")]
    print("[summary_cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=THIS_DIR, check=True)


def parse_args():
    p = argparse.ArgumentParser()
    n_gpus = torch.cuda.device_count()
    p.add_argument("--nproc_per_node", type=int, default=n_gpus)
    p.add_argument("--master_port", type=int, default=29600)
    p.add_argument("--start_from", type=str, default="church_snow1")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    features = get_church_features()
    if args.start_from not in features:
        raise SystemExit(f"start_from not found in church feature list: {args.start_from}")

    start_idx = features.index(args.start_from)
    selected = features[start_idx:]
    env = os.environ.copy()

    print(f"[batch] total church features ending with 1: {len(features)}")
    print(f"[batch] starting from: {args.start_from}")
    print(f"[batch] selected count: {len(selected)}")
    print(f"[batch] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'all')}")

    progress = tqdm(selected, desc="features", unit="feature")
    total = len(selected)
    try:
        for idx, feature in enumerate(progress, start=1):
            progress.set_description(f"feature {idx}/{total}: {feature}")
            if has_existing_result(feature):
                print(f"[skip] {feature}: out/{feature} already has result")
                continue
            cmd = build_command(args, feature)
            print(f"[run] {feature} ({idx}/{total})")
            print("[cmd]", " ".join(cmd))
            if args.dry_run:
                continue
            subprocess.run(cmd, cwd=THIS_DIR, env=env, check=True)
    finally:
        progress.close()

    if args.dry_run:
        print("[batch] dry run finished; skip summary.py")
        return
    print("[batch] all done, running summary.py")
    run_summary()


if __name__ == "__main__":
    main()
