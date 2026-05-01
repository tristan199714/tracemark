import os
from pathlib import Path
from typing import Dict, List

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from .CelebA_HQ_dataset import MultiResolutionDataset


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
    candidates = [
        root,
        root / "CUB_200_2011",
    ]
    for candidate in candidates:
        if (candidate / "images.txt").exists() and (candidate / "train_test_split.txt").exists():
            return candidate
    raise FileNotFoundError(
        f"CUB root not found under {data_root}. Expected images.txt and train_test_split.txt."
    )


def _has_lmdb_cache(data_root: str) -> bool:
    root = Path(data_root)
    return (root / "LMDB_train").exists() and (root / "LMDB_test").exists()


class CUBDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        transform=None,
        split: str = "train",
        img_size: int = 256,
        crop_to_bbox: bool = True,
        bbox_pad_frac: float = 0.10,
    ):
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.dataset_root = _resolve_cub_root(data_root)
        self.transform = transform
        self.img_size = int(img_size)
        self.crop_to_bbox = bool(crop_to_bbox)
        self.bbox_pad_frac = float(bbox_pad_frac)
        self.records = self._load_records(split=split)

    def _load_records(self, split: str):
        image_paths = _read_id_map(self.dataset_root / "images.txt")
        split_map = _read_id_int_map(self.dataset_root / "train_test_split.txt")
        bbox_map = _read_id_float_map(self.dataset_root / "bounding_boxes.txt")

        want_train = 1 if split == "train" else 0
        records = []
        for image_id in sorted(image_paths):
            if split_map.get(image_id) != want_train:
                continue
            rel_path = image_paths[image_id]
            bbox = bbox_map.get(image_id)
            if bbox is None:
                continue
            records.append(
                {
                    "image_id": image_id,
                    "path": self.dataset_root / "images" / rel_path,
                    "bbox": bbox,
                }
            )
        return records

    def _crop_bbox(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        if not self.crop_to_bbox:
            return image

        x, y, w, h = bbox
        width, height = image.size
        side = max(w, h) * (1.0 + self.bbox_pad_frac * 2.0)
        cx = x + w / 2.0
        cy = y + h / 2.0

        left = max(0.0, cx - side / 2.0)
        top = max(0.0, cy - side / 2.0)
        right = min(float(width), cx + side / 2.0)
        bottom = min(float(height), cy + side / 2.0)

        if right <= left or bottom <= top:
            return image

        return image.crop((int(left), int(top), int(right), int(bottom)))

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record["path"]).convert("RGB")
        image = self._crop_bbox(image, record["bbox"])
        image = image.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.records)


def get_cub_dataset(data_root, config):
    train_transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])
    test_transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])

    crop_to_bbox = bool(getattr(config.data, "crop_to_bbox", True))
    bbox_pad_frac = float(getattr(config.data, "bbox_pad_frac", 0.10))

    if _has_lmdb_cache(data_root):
        train_dataset = MultiResolutionDataset(
            os.path.join(data_root, "LMDB_train"),
            train_transform,
            config.data.image_size,
        )
        test_dataset = MultiResolutionDataset(
            os.path.join(data_root, "LMDB_test"),
            test_transform,
            config.data.image_size,
        )
        return train_dataset, test_dataset

    train_dataset = CUBDataset(
        data_root,
        transform=train_transform,
        split="train",
        img_size=config.data.image_size,
        crop_to_bbox=crop_to_bbox,
        bbox_pad_frac=bbox_pad_frac,
    )
    test_dataset = CUBDataset(
        data_root,
        transform=test_transform,
        split="test",
        img_size=config.data.image_size,
        crop_to_bbox=crop_to_bbox,
        bbox_pad_frac=bbox_pad_frac,
    )
    return train_dataset, test_dataset
