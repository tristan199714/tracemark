import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = THIS_DIR / "out"
DEFAULT_REPORT_DIR = THIS_DIR / "report"

METRIC_KEYS = [
    "id_acc_orig",
    "id_acc_pre",
    "id_acc_wm",
    "verify_pre_auc",
    "verify_pre_acc",
    "verify_pre_tpr1",
    "verify_orig_auc",
    "verify_orig_acc",
    "verify_orig_tpr1",
    "fid_wm_pre",
    "fid_wm_orig",
    "clip_wm_pre",
    "clip_wm_orig",
    "is_wm_mean",
]


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_feature_run(feature_dir: Path) -> Optional[Path]:
    if (feature_dir / "summary.json").exists():
        return feature_dir
    runs = sorted(
        [p for p in feature_dir.glob("run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


def _classify_dataset(feature: str) -> str:
    if feature.startswith("dog_"):
        return "AFHQ_Dog"
    if feature.startswith("church_"):
        return "LSUN_Church"
    if feature.startswith("bedroom_"):
        return "LSUN_Bedroom"
    return "CelebA_HQ"


def _flatten_metrics(feature: str, run_dir: Path) -> Dict[str, object]:
    summary_path = _first_existing([run_dir / "summary.json"])
    if summary_path is None:
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    summary = _load_json(summary_path)
    eval_path = _first_existing([run_dir / "eval_auto.json", run_dir / "eval_manual.json"])
    eval_data = _load_json(eval_path) if eval_path is not None else {}
    args = summary.get("args", {})
    quality = eval_data.get("quality", {})
    wm_pre = quality.get("wm_vs_pre", {})
    wm_orig = quality.get("wm_vs_orig", {})
    is_wm = quality.get("is_wm", {})
    id_acc = eval_data.get("id_acc", {})
    counts = eval_data.get("counts", {})
    verify_pre = eval_data.get("verify_wm_vs_pre", {})
    verify_orig = eval_data.get("verify_wm_vs_orig", {})

    return {
        "feature": feature,
        "dataset": _classify_dataset(feature),
        "run_dir": str(run_dir.relative_to(THIS_DIR)),
        "run_name": summary.get("run_name", run_dir.name),
        "instruction_text": summary.get("instruction", ""),
        "num_user": args.get("num_user", summary.get("num_user", "")),
        "embed_dim": args.get("embed_dim", summary.get("embed_dim", "")),
        "n_iter": args.get("n_iter", ""),
        "n_train_img": args.get("n_train_img", ""),
        "bs_train": args.get("bs_train", ""),
        "lr": args.get("lr", ""),
        "wm_strength": args.get("wm_strength", ""),
        "sim_lambda": args.get("sim_lambda", ""),
        "carr_lambda": args.get("carr_lambda", ""),
        "cons_lambda": args.get("cons_lambda", ""),
        "out_decode_lambda": args.get("out_decode_lambda", ""),
        "ip2p_steps": args.get("ip2p_steps", ""),
        "count_orig": counts.get("orig", ""),
        "count_pre": counts.get("pre", ""),
        "count_wm": counts.get("wm", ""),
        "id_acc_orig": id_acc.get("orig", ""),
        "id_acc_pre": id_acc.get("pre", ""),
        "id_acc_wm": id_acc.get("wm", ""),
        "verify_pre_auc": verify_pre.get("auc", ""),
        "verify_pre_acc": verify_pre.get("acc", ""),
        "verify_pre_tpr1": verify_pre.get("tpr_at_1pct_fpr", ""),
        "verify_orig_auc": verify_orig.get("auc", ""),
        "verify_orig_acc": verify_orig.get("acc", ""),
        "verify_orig_tpr1": verify_orig.get("tpr_at_1pct_fpr", ""),
        "fid_wm_pre": wm_pre.get("fid", ""),
        "fid_wm_orig": wm_orig.get("fid", ""),
        "clip_wm_pre": wm_pre.get("clip_sim", ""),
        "clip_wm_orig": wm_orig.get("clip_sim", ""),
        "is_wm_mean": is_wm.get("mean", ""),
        "is_wm_std": is_wm.get("std", ""),
    }


def build_rows(out_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    feature_dirs = sorted([p for p in out_root.iterdir() if p.is_dir()])
    progress = tqdm(feature_dirs, desc="summary", unit="feature")
    total = len(feature_dirs)
    for idx, feature_dir in enumerate(progress, start=1):
        progress.set_description(f"summary {idx}/{total}: {feature_dir.name}")
        run_dir = _resolve_feature_run(feature_dir)
        if run_dir is None:
            continue
        try:
            rows.append(_flatten_metrics(feature_dir.name, run_dir))
        except FileNotFoundError:
            continue
    return rows


def build_dataset_avg(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        buckets[row["dataset"]].append(row)

    avg_rows = []
    for dataset in ["CelebA_HQ", "AFHQ_Dog", "LSUN_Church", "LSUN_Bedroom"]:
        group = buckets.get(dataset, [])
        if not group:
            continue
        entry: Dict[str, object] = {"dataset": dataset, "num_features": len(group)}
        for key in METRIC_KEYS:
            vals = [float(r[key]) for r in group if r.get(key, "") != "" and r[key] is not None]
            entry[key] = sum(vals) / len(vals) if vals else ""
        avg_rows.append(entry)

    if len(avg_rows) > 1:
        all_rows = [r for r in rows if r.get("dataset", "") != ""]
        if all_rows:
            entry = {"dataset": "ALL", "num_features": len(all_rows)}
            for key in METRIC_KEYS:
                vals = [float(r[key]) for r in all_rows if r.get(key, "") != "" and r[key] is not None]
                entry[key] = sum(vals) / len(vals) if vals else ""
            avg_rows.append(entry)

    return avg_rows


def group_rows_by_dataset(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        dataset = str(row.get("dataset", "")).strip()
        if not dataset:
            continue
        buckets[dataset].append(row)
    return dict(buckets)


def dataset_report_stem(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["feature"] if not rows else list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--report_dir", type=str, default=str(DEFAULT_REPORT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(out_root)

    detail_csv = report_dir / "detail.csv"
    write_csv(rows, detail_csv)
    print(f"[summary] wrote {len(rows)} feature rows to {detail_csv}")

    avg_rows = build_dataset_avg(rows)
    avg_csv = report_dir / "dataset_avg.csv"
    write_csv(avg_rows, avg_csv)
    print(f"[summary] wrote {len(avg_rows)} dataset rows to {avg_csv}")

    by_dataset = group_rows_by_dataset(rows)
    for dataset, dataset_rows in by_dataset.items():
        stem = dataset_report_stem(dataset)
        dataset_detail_csv = report_dir / f"detail_{stem}.csv"
        write_csv(dataset_rows, dataset_detail_csv)
        print(f"[summary] wrote {len(dataset_rows)} feature rows to {dataset_detail_csv}")

        dataset_avg_rows = build_dataset_avg(dataset_rows)
        dataset_avg_csv = report_dir / f"dataset_avg_{stem}.csv"
        write_csv(dataset_avg_rows, dataset_avg_csv)
        print(f"[summary] wrote {len(dataset_avg_rows)} dataset rows to {dataset_avg_csv}")

    for r in avg_rows:
        ds = r["dataset"]
        n = r["num_features"]
        acc = r.get("id_acc_wm", "")
        auc = r.get("verify_pre_auc", "")
        fid = r.get("fid_wm_pre", "")
        clip = r.get("clip_wm_pre", "")
        acc_s = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        auc_s = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        fid_s = f"{fid:.2f}" if isinstance(fid, float) else str(fid)
        clip_s = f"{clip:.4f}" if isinstance(clip, float) else str(clip)
        print(f"  {ds} (n={n}): id_acc_wm={acc_s} verify_pre_auc={auc_s} fid_wm_pre={fid_s} clip_wm_pre={clip_s}")


if __name__ == "__main__":
    main()
