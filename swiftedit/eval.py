import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

THIS_DIR = Path(__file__).resolve().parent
SHARED_ROOT = THIS_DIR.parent / "instructPix2Pix"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_SHARED_EVAL = _load_module("shared_instruct_eval", SHARED_ROOT / "eval.py")
_SHARED_MODELS = _load_module("shared_instruct_models", SHARED_ROOT / "models.py")

compute_roc_metrics = _SHARED_EVAL.compute_roc_metrics
list_images = _SHARED_EVAL.list_images
safe_clip_similarity = _SHARED_EVAL.safe_clip_similarity
safe_fid = _SHARED_EVAL.safe_fid
safe_inception = _SHARED_EVAL.safe_inception
RetrievalDetector = _SHARED_MODELS.RetrievalDetector


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
    for record in records:
        fn = record.get("filename")
        uid = record.get("uid")
        if fn is None or uid is None:
            continue
        if "saved" in record and not bool(record.get("saved", False)):
            continue
        uid_by_filename[str(fn)] = int(uid)

    if not uid_by_filename:
        for path in list_images(image_root / "wm"):
            if path.stem.isdigit():
                uid_by_filename[path.name] = int(path.stem) % num_user

    tfm = transforms.Compose(
        [
            transforms.Resize((int(saved_args.get("resolution", 256)), int(saved_args.get("resolution", 256)))),
            transforms.ToTensor(),
        ]
    )

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

    id_acc = {
        split: float(split_correct[split] / max(1, split_total[split]))
        for split in ["orig", "pre", "wm"]
    }

    auc_pre, acc_pre, tpr1_pre = compute_roc_metrics(split_scores["wm"], split_scores["pre"])
    auc_orig, acc_orig, tpr1_orig = compute_roc_metrics(split_scores["wm"], split_scores["orig"])

    summary = {
        "id_acc": id_acc,
        "counts": {key: int(value) for key, value in split_total.items()},
        "verify_wm_vs_pre": {"auc": auc_pre, "acc": acc_pre, "tpr_at_1pct_fpr": tpr1_pre},
        "verify_wm_vs_orig": {"auc": auc_orig, "acc": acc_orig, "tpr_at_1pct_fpr": tpr1_orig},
        "quality": {"wm_vs_pre": {}, "wm_vs_orig": {}, "is_wm": {}},
        "warnings": [],
    }

    if not args.no_quality:
        fid_pre, err = safe_fid(image_root / "pre", image_root / "wm", device)
        summary["quality"]["wm_vs_pre"]["fid"] = fid_pre
        if err:
            summary["warnings"].append(f"FID wm_vs_pre failed: {err}")

        fid_orig, err = safe_fid(image_root / "orig", image_root / "wm", device)
        summary["quality"]["wm_vs_orig"]["fid"] = fid_orig
        if err:
            summary["warnings"].append(f"FID wm_vs_orig failed: {err}")

        is_mean, is_std, err = safe_inception(image_root / "wm", device)
        summary["quality"]["is_wm"]["mean"] = is_mean
        summary["quality"]["is_wm"]["std"] = is_std
        if err:
            summary["warnings"].append(f"IS wm failed: {err}")

        clip_pre, err = safe_clip_similarity(image_root / "pre", image_root / "wm", device)
        summary["quality"]["wm_vs_pre"]["clip_sim"] = clip_pre
        if err:
            summary["warnings"].append(f"CLIP wm_vs_pre failed: {err}")

        clip_orig, err = safe_clip_similarity(image_root / "orig", image_root / "wm", device)
        summary["quality"]["wm_vs_orig"]["clip_sim"] = clip_orig
        if err:
            summary["warnings"].append(f"CLIP wm_vs_orig failed: {err}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[eval] wrote {out_json}")


if __name__ == "__main__":
    main()
