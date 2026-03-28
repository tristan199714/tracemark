import argparse
import json
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train import Pix2PixTurboEmbedMarkTrainer


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, local_rank


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def _slug(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "exp"


def _f(val: float) -> str:
    return f"{val:g}".replace(".", "p")


def build_run_name(args) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"run_{ts}_ins-{_slug(args.instruction)}"
        f"_seed{args.seed}"
        f"_u{args.num_user}_it{args.n_iter}_n{args.n_train_img}_bs{args.bs_train}"
        f"_wm{_f(args.wm_strength)}_id{_f(args.id_lambda)}"
        f"_sim{_f(args.sim_lambda)}_car{_f(args.carr_lambda)}"
        f"_cons{_f(args.cons_lambda)}"
        f"_out{_f(args.out_decode_lambda)}"
        f"_wh{args.writer_hidden}_wb{args.writer_blocks}"
        f"_ud{args.user_dim}_ed{args.embed_dim}"
        f"_lr{_f(args.lr)}"
    )


def prepare_run_paths(args):
    out_dir = Path(args.exp) / "out" / args.instruction
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = build_run_name(args)
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_name, run_dir, run_dir / "train.log"


def sync_run_paths(args):
    if args.distributed and dist.is_initialized():
        payload = [None, None, None]
        if args.rank == 0:
            run_name, run_dir, log_path = prepare_run_paths(args)
            payload = [run_name, str(run_dir), str(log_path)]
        dist.broadcast_object_list(payload, src=0)
        run_name, run_dir, log_path = payload[0], Path(payload[1]), Path(payload[2])
        return run_name, run_dir, log_path
    return prepare_run_paths(args)


def dump_run_metadata(args, cfg, run_dir: Path):
    payload = {
        "argv": sys.argv,
        "args": vars(args).copy(),
        "config": cfg,
    }
    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="../instructPix2Pix/configs/celeba.yml")
    p.add_argument("--exp", type=str, default=str(THIS_DIR))
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--turbo_model_id", type=str, default="stabilityai/sd-turbo")
    p.add_argument("--turbo_lora_path", type=str, default="")
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--instruction", type=str, default="beards1")
    p.add_argument("--num_user", type=int, default=15)

    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--turbo_steps", type=int, default=1)
    p.add_argument("--turbo_strength", type=float, default=0.55)
    p.add_argument("--guidance_scale", type=float, default=0.0)
    p.add_argument("--fp16", type=int, default=0)

    p.add_argument("--n_iter", type=int, default=80)
    p.add_argument("--n_train_img", type=int, default=200)
    p.add_argument("--bs_train", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--wm_strength", type=float, default=0.12)
    p.add_argument("--warmup_no_edit_iters", type=int, default=10)
    p.add_argument("--edit_ramp_iters", type=int, default=20)

    p.add_argument("--writer_hidden", type=int, default=96)
    p.add_argument("--writer_blocks", type=int, default=4)
    p.add_argument("--user_dim", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=128)

    p.add_argument("--id_lambda", type=float, default=1.0)
    p.add_argument("--sim_lambda", type=float, default=0.5)
    p.add_argument("--carr_lambda", type=float, default=0.2)
    p.add_argument("--cons_lambda", type=float, default=0.5)
    p.add_argument("--wm_decode_lambda", type=float, default=1.0)
    p.add_argument("--out_decode_lambda", type=float, default=1.0)

    p.add_argument("--save_images", type=int, default=1)
    p.add_argument("--clean_image_out", type=int, default=1)
    p.add_argument("--auto_eval", type=int, default=1)
    p.add_argument("--batch_pbar", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    args.distributed = distributed
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    set_seed(args.seed + rank)

    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    log_fp = None
    run_name, run_dir, log_path = sync_run_paths(args)
    args.run_name = run_name
    args.run_dir = str(run_dir)
    if rank == 0:
        log_fp = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = TeeStream(stdout_orig, log_fp)
        sys.stderr = TeeStream(stderr_orig, log_fp)
        print(f"[Pix2PixTurbo] log_file: {log_path}")
        print(f"[Pix2PixTurbo] run_dir: {run_dir}")

    try:
        cfg = yaml.safe_load(open(args.config))
        if rank == 0:
            dump_run_metadata(args, cfg, run_dir)
            print("[Pix2PixTurbo] argv:", " ".join(sys.argv))
            print("[Pix2PixTurbo] args:", json.dumps(vars(args), ensure_ascii=False, sort_keys=True))
        trainer = Pix2PixTurboEmbedMarkTrainer(args, cfg)
        trainer.run()
    finally:
        if distributed and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
        if log_fp is not None:
            sys.stdout = stdout_orig
            sys.stderr = stderr_orig
            log_fp.close()


if __name__ == "__main__":
    main()
