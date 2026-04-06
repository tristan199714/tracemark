import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from models import RetrievalDetector


def list_images(root: Path) -> List[Path]:
    paths = []
    for dp, _, fs in os.walk(str(root)):
        for f in fs:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(Path(dp) / f)
    return sorted(paths)


def compute_roc_metrics(scores_pos: np.ndarray, scores_neg: np.ndarray):
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    if scores_pos.size == 0 or scores_neg.size == 0:
        return 0.0, 0.0, 0.0
    s = np.concatenate([scores_pos, scores_neg], axis=0)
    y = np.concatenate([np.ones_like(scores_pos), np.zeros_like(scores_neg)], axis=0)
    thr = np.unique(s)[::-1]
    fpr_list = [0.0]
    tpr_list = [0.0]
    for t in thr:
        pred = (s >= t).astype(np.int32)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        tpr_list.append(tp / max(1, tp + fn))
        fpr_list.append(fp / max(1, fp + tn))
    fpr_list.append(1.0)
    tpr_list.append(1.0)
    fpr = np.array(fpr_list, dtype=float)
    tpr = np.array(tpr_list, dtype=float)
    order = np.argsort(fpr)
    auc_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(auc_fn(tpr[order], fpr[order]))
    acc = float(np.max(1.0 - (fpr + (1.0 - tpr)) / 2.0))
    idx = np.where(fpr < 0.01)[0]
    tpr1 = float(tpr[idx[-1]]) if idx.size else 0.0
    return auc, acc, tpr1


def safe_fid(folder_a: Path, folder_b: Path, device: str):
    try:
        from pytorch_fid import fid_score

        value = fid_score.calculate_fid_given_paths(
            [str(folder_a), str(folder_b)],
            batch_size=32,
            device=device,
            dims=2048,
        )
        return float(value), None
    except Exception as e:
        return None, str(e)


def safe_inception(folder: Path, device: str):
    try:
        from torchvision.models import inception_v3
    except Exception as e:
        return None, None, f"torchvision inception import failed: {e}"

    model = None
    try:
        from torchvision.models import Inception_V3_Weights

        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device).eval()
    except Exception:
        try:
            model = inception_v3(pretrained=True, transform_input=False).to(device).eval()
        except Exception as e:
            return None, None, str(e)

    class _ImageFolderDataset(torch.utils.data.Dataset):
        def __init__(self, folder_path: Path):
            self.paths = list_images(folder_path)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((299, 299)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)

    ds = _ImageFolderDataset(folder)
    if len(ds) == 0:
        return 0.0, 0.0, None
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds.append(F.softmax(logits, dim=1).cpu())
    preds = torch.cat(preds, dim=0)
    n = preds.size(0)
    splits = max(1, min(10, n))
    split_size = max(1, n // splits)
    eps = 1e-8
    scores = []
    for i in range(splits):
        start = i * split_size
        end = (i + 1) * split_size if i < splits - 1 else n
        part = preds[start:end]
        if part.numel() == 0:
            continue
        py = torch.mean(part, dim=0, keepdim=True)
        kl = part * (torch.log(part + eps) - torch.log(py + eps))
        score = torch.exp(torch.mean(torch.sum(kl, dim=1)))
        scores.append(float(score.item()))
    if not scores:
        return 0.0, 0.0, None
    return float(np.mean(scores)), float(np.std(scores)), None


def safe_clip_similarity(folder_a: Path, folder_b: Path, device: str, clip_model: str = "openai/clip-vit-base-patch32"):
    try:
        from transformers import CLIPModel
    except Exception as e:
        return None, str(e)
    try:
        model = CLIPModel.from_pretrained(clip_model).to(device).eval()
    except Exception as e:
        return None, str(e)

    def rel(p: Path, root: Path):
        return os.path.relpath(str(p), str(root)).replace("\\", "/")

    a = list_images(folder_a)
    b = list_images(folder_b)
    bmap = {rel(p, folder_b): p for p in b}
    keys = [rel(p, folder_a) for p in a if rel(p, folder_a) in bmap]
    if not keys:
        return 0.0, None

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275)),
        ]
    )
    sims = []
    with torch.no_grad():
        for k in keys:
            pa = folder_a / k
            pb = bmap[k]
            ia = Image.open(pa).convert("RGB")
            ib = Image.open(pb).convert("RGB")
            xa = preprocess(ia).unsqueeze(0).to(device)
            xb = preprocess(ib).unsqueeze(0).to(device)
            fa = model.get_image_features(pixel_values=xa)
            fb = model.get_image_features(pixel_values=xb)
            sims.append(float(F.cosine_similarity(fa, fb).item()))
    return float(np.mean(sims)) if sims else 0.0, None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(THIS_DIR / "out" / "beards1" / "checkpoint.pt"))
    p.add_argument("--image_root", type=str, default=str(THIS_DIR / "out" / "beards1"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_quality", action="store_true")
    p.add_argument("--out_json", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu"

    ckpt_path = Path(args.ckpt)
    image_root = Path(args.image_root)
    out_json = Path(args.out_json) if args.out_json else ckpt_path.parent / "eval_manual.json"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_args = ckpt.get("args", {})
    num_user = int(saved_args.get("num_user", 15))
    embed_dim = int(saved_args.get("embed_dim", 128))

    det = RetrievalDetector(num_user=num_user, embed_dim=embed_dim).to(device).eval()
    det.load_state_dict(ckpt["detector"], strict=True)
    det_dtype = next(det.parameters()).dtype

    records = ckpt.get("records", [])
    uid_by_filename: Dict[str, int] = {}
    for r in records:
        fn = r.get("filename")
        uid = r.get("uid")
        if fn is None or uid is None:
            continue
        if "saved" in r and not bool(r.get("saved", False)):
            continue
        uid_by_filename[str(fn)] = int(uid)

    # Fallback for old checkpoints that do not store filename/saved.
    if not uid_by_filename:
        for p in list_images(image_root / "wm"):
            stem = p.stem
            if stem.isdigit():
                uid_by_filename[p.name] = int(stem) % num_user

    tfm = transforms.Compose([
        transforms.Resize((int(saved_args.get("resolution", 256)), int(saved_args.get("resolution", 256)))),
        transforms.ToTensor(),
    ])

    split_scores = {"orig": [], "pre": [], "wm": []}
    split_correct = {"orig": 0.0, "pre": 0.0, "wm": 0.0}
    split_total = {"orig": 0, "pre": 0, "wm": 0}

    with torch.no_grad():
        for split in ["orig", "pre", "wm"]:
            split_dir = image_root / split
            for img_path in sorted(split_dir.glob("*.png")):
                fn = img_path.name
                if fn not in uid_by_filename:
                    continue
                uid = int(uid_by_filename[fn])
                img = Image.open(img_path).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device) * 2 - 1
                if x.dtype != det_dtype:
                    x = x.to(det_dtype)
                _, logits = det(x)
                probs = F.softmax(logits, dim=1)
                pred = int(torch.argmax(logits, dim=1).item())
                conf = float(torch.max(probs, dim=1).values.item())
                split_scores[split].append(conf)
                split_correct[split] += float(pred == uid)
                split_total[split] += 1

    id_acc = {}
    for split in ["orig", "pre", "wm"]:
        id_acc[split] = float(split_correct[split] / max(1, split_total[split]))

    auc_pre, acc_pre, tpr1_pre = compute_roc_metrics(
        np.asarray(split_scores["wm"], dtype=float),
        np.asarray(split_scores["pre"], dtype=float),
    )
    auc_orig, acc_orig, tpr1_orig = compute_roc_metrics(
        np.asarray(split_scores["wm"], dtype=float),
        np.asarray(split_scores["orig"], dtype=float),
    )

    result = {
        "id_acc": id_acc,
        "counts": {k: int(v) for k, v in split_total.items()},
        "verify_wm_vs_pre": {"auc": auc_pre, "acc": acc_pre, "tpr_at_1pct_fpr": tpr1_pre},
        "verify_wm_vs_orig": {"auc": auc_orig, "acc": acc_orig, "tpr_at_1pct_fpr": tpr1_orig},
    }

    if not args.no_quality:
        quality = {"wm_vs_pre": {}, "wm_vs_orig": {}, "is_wm": {}}
        warnings = []
        fid_pre, err = safe_fid(image_root / "pre", image_root / "wm", device)
        quality["wm_vs_pre"]["fid"] = fid_pre
        if err:
            warnings.append(f"FID wm_vs_pre failed: {err}")
        fid_orig, err = safe_fid(image_root / "orig", image_root / "wm", device)
        quality["wm_vs_orig"]["fid"] = fid_orig
        if err:
            warnings.append(f"FID wm_vs_orig failed: {err}")

        is_mean, is_std, err = safe_inception(image_root / "wm", device)
        quality["is_wm"]["mean"] = is_mean
        quality["is_wm"]["std"] = is_std
        if err:
            warnings.append(f"IS wm failed: {err}")

        clip_pre, err = safe_clip_similarity(image_root / "pre", image_root / "wm", device)
        quality["wm_vs_pre"]["clip_sim"] = clip_pre
        if err:
            warnings.append(f"CLIP wm_vs_pre failed: {err}")
        clip_orig, err = safe_clip_similarity(image_root / "orig", image_root / "wm", device)
        quality["wm_vs_orig"]["clip_sim"] = clip_orig
        if err:
            warnings.append(f"CLIP wm_vs_orig failed: {err}")
        result["quality"] = quality
        result["warnings"] = warnings

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[EmbedMark eval] saved: {out_json}")
    print(
        "[EmbedMark eval] "
        f"id_acc(orig/pre/wm)="
        f"{result['id_acc']['orig']:.4f}/"
        f"{result['id_acc']['pre']:.4f}/"
        f"{result['id_acc']['wm']:.4f} "
        f"wm_vs_pre(auc/acc/tpr1)="
        f"{result['verify_wm_vs_pre']['auc']:.4f}/"
        f"{result['verify_wm_vs_pre']['acc']:.4f}/"
        f"{result['verify_wm_vs_pre']['tpr_at_1pct_fpr']:.4f} "
        f"wm_vs_orig(auc/acc/tpr1)="
        f"{result['verify_wm_vs_orig']['auc']:.4f}/"
        f"{result['verify_wm_vs_orig']['acc']:.4f}/"
        f"{result['verify_wm_vs_orig']['tpr_at_1pct_fpr']:.4f}"
    )
    if "quality" in result:
        q = result["quality"]
        print(
            "[EmbedMark eval] "
            f"fid(wm,pre)={q['wm_vs_pre'].get('fid')} "
            f"fid(wm,orig)={q['wm_vs_orig'].get('fid')} "
            f"is(wm)={q['is_wm'].get('mean')}+-{q['is_wm'].get('std')} "
            f"clip(wm,pre)={q['wm_vs_pre'].get('clip_sim')} "
            f"clip(wm,orig)={q['wm_vs_orig'].get('clip_sim')}"
        )
        for w in result.get("warnings", []):
            print(f"[EmbedMark eval warning] {w}")


if __name__ == "__main__":
    main()
