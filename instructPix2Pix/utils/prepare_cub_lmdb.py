"""
Prepare cropped/resized LMDB caches for CUB-200-2011.

This matches the LMDB layout expected by MultiResolutionDataset:
  <out>/LMDB_train
  <out>/LMDB_test
with keys like "256-00000" and a "length" entry.
"""

import argparse
from functools import partial
from io import BytesIO
import multiprocessing
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

import lmdb
from PIL import Image
from tqdm import tqdm


def _read_id_map(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, value = line.split(" ", 1)
            mapping[int(idx_str)] = value.strip()
    return mapping


def _read_id_float_map(path: Path) -> Dict[int, List[float]]:
    mapping: Dict[int, List[float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            mapping[int(parts[0])] = [float(v) for v in parts[1:]]
    return mapping


def _read_id_int_map(path: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            mapping[int(parts[0])] = int(parts[1])
    return mapping


def _resolve_cub_root(data_root: str) -> Path:
    root = Path(data_root)
    candidates = [root, root / "CUB_200_2011"]
    for candidate in candidates:
        if (candidate / "images.txt").exists() and (candidate / "train_test_split.txt").exists():
            return candidate
    raise FileNotFoundError(
        f"CUB root not found under {data_root}. Expected images.txt and train_test_split.txt."
    )


def _crop_bbox(image: Image.Image, bbox: List[float], pad_frac: float) -> Image.Image:
    x, y, w, h = bbox
    width, height = image.size
    side = max(w, h) * (1.0 + pad_frac * 2.0)
    cx = x + w / 2.0
    cy = y + h / 2.0

    left = max(0.0, cx - side / 2.0)
    top = max(0.0, cy - side / 2.0)
    right = min(float(width), cx + side / 2.0)
    bottom = min(float(height), cy + side / 2.0)

    if right <= left or bottom <= top:
        return image
    return image.crop((int(left), int(top), int(right), int(bottom)))


def _resize_and_convert(img: Image.Image, size: int, resample, quality: int = 100) -> bytes:
    img = img.resize((size, size), resample=resample)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    return buffer.getvalue()


def _resize_worker(record: Tuple[int, Dict[str, object]], sizes, resample, bbox_pad_frac: float):
    index, item = record
    img = Image.open(item["path"]).convert("RGB")
    img = _crop_bbox(img, item["bbox"], bbox_pad_frac)
    outputs = [_resize_and_convert(img, size, resample) for size in sizes]
    return index, outputs


def _build_records(cub_root: Path, split: str):
    image_paths = _read_id_map(cub_root / "images.txt")
    split_map = _read_id_int_map(cub_root / "train_test_split.txt")
    bbox_map = _read_id_float_map(cub_root / "bounding_boxes.txt")
    want_train = 1 if split == "train" else 0

    records = []
    for image_id in sorted(image_paths):
        if split_map.get(image_id) != want_train:
            continue
        bbox = bbox_map.get(image_id)
        if bbox is None:
            continue
        records.append(
            {
                "image_id": image_id,
                "path": cub_root / "images" / image_paths[image_id],
                "bbox": bbox,
            }
        )
    return records


def _prepare_split(env_path: Path, records, n_worker: int, sizes, resample, bbox_pad_frac: float):
    resize_fn = partial(
        _resize_worker,
        sizes=sizes,
        resample=resample,
        bbox_pad_frac=bbox_pad_frac,
    )
    indexed_records = [(i, record) for i, record in enumerate(records)]
    total = 0

    with lmdb.open(str(env_path), map_size=32 * (1024 ** 3), readahead=False) as env:
        with multiprocessing.Pool(n_worker) as pool:
            for i, imgs in tqdm(pool.imap_unordered(resize_fn, indexed_records), total=len(indexed_records), desc=env_path.name):
                with env.begin(write=True) as txn:
                    for size, img in zip(sizes, imgs):
                        key = f"{size}-{str(i).zfill(5)}".encode("utf-8")
                        txn.put(key, img)
                total += 1

        with env.begin(write=True) as txn:
            txn.put(b"length", str(total).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/Sheldon/diffusion_data/cub_200_2011")
    parser.add_argument("--sizes", type=str, default="256")
    parser.add_argument("--n_worker", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    parser.add_argument("--bbox_pad_frac", type=float, default=0.10)
    parser.add_argument("--resample", type=str, default="lanczos", choices=["lanczos", "bilinear"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    out_root = Path(args.data_root)
    cub_root = _resolve_cub_root(args.data_root)
    out_train = out_root / "LMDB_train"
    out_test = out_root / "LMDB_test"

    for out_path in [out_train, out_test]:
        if out_path.exists():
            if not args.overwrite:
                raise FileExistsError(f"{out_path} already exists. Re-run with --overwrite to rebuild it.")
            shutil.rmtree(out_path)

    print(f"[prepare_cub_lmdb] cub_root={cub_root}")
    print(f"[prepare_cub_lmdb] out_root={out_root}")
    print(f"[prepare_cub_lmdb] sizes={sizes} n_worker={args.n_worker} bbox_pad_frac={args.bbox_pad_frac}")

    train_records = _build_records(cub_root, split="train")
    test_records = _build_records(cub_root, split="test")
    print(f"[prepare_cub_lmdb] train_records={len(train_records)} test_records={len(test_records)}")

    _prepare_split(out_train, train_records, args.n_worker, sizes, resample, args.bbox_pad_frac)
    _prepare_split(out_test, test_records, args.n_worker, sizes, resample, args.bbox_pad_frac)

    print(f"[prepare_cub_lmdb] wrote {out_train}")
    print(f"[prepare_cub_lmdb] wrote {out_test}")


if __name__ == "__main__":
    main()
