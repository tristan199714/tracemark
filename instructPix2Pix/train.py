import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.data_utils import get_dataset  # noqa: E402
from utils.text_dic import INSTRUCTION_MAP  # noqa: E402
from models import RetrievalDetector, UserConditionedWriter  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_pil(x01: torch.Tensor) -> Image.Image:
    arr = (x01.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def list_images(root: Path):
    paths = []
    for dp, _, fs in os.walk(str(root)):
        for f in fs:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(Path(dp) / f)
    return sorted(paths)


class IP2PEmbedMarkTrainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.rank = int(getattr(args, "rank", 0))
        self.world_size = int(getattr(args, "world_size", 1))
        self.local_rank = int(getattr(args, "local_rank", 0))
        self.distributed = bool(getattr(args, "distributed", False))
        self.is_main = self.rank == 0

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
        set_seed(args.seed + self.rank)

        self.writer = UserConditionedWriter(
            num_user=args.num_user,
            user_dim=args.user_dim,
            hidden=args.writer_hidden,
            blocks=args.writer_blocks,
        ).to(self.device)
        self.detector = RetrievalDetector(
            num_user=args.num_user,
            embed_dim=args.embed_dim,
        ).to(self.device)

        if self.distributed:
            if self.device.type == "cuda":
                self.writer = DDP(
                    self.writer,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                )
                self.detector = DDP(
                    self.detector,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False,
                )
            else:
                self.writer = DDP(self.writer, find_unused_parameters=False)
                self.detector = DDP(self.detector, find_unused_parameters=False)

        self.optimizer = torch.optim.Adam(
            list(self.writer.parameters()) + list(self.detector.parameters()),
            lr=args.lr,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.n_iter,
            eta_min=args.lr * 0.05,
        )

        self.pipe = self._build_ip2p()
        self.instruction = self._resolve_instruction(args.instruction)

        train_ds, _ = get_dataset(config["data"]["dataset"], config=config)
        self.loader, self.sampler = self._build_loader(train_ds)

        default_out_root = Path(args.exp) / "out" / args.instruction
        self.out_root = Path(getattr(args, "run_dir", default_out_root))
        self.image_out_root = self.out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        for k in ["orig", "pre", "wm"]:
            d = self.image_out_root / k
            d.mkdir(parents=True, exist_ok=True)
            if self.is_main and int(getattr(self.args, "clean_image_out", 1)) == 1:
                for p in d.glob("*.png"):
                    p.unlink()
        if self.distributed and dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _unwrap_module(module):
        return module.module if isinstance(module, DDP) else module

    def _reduce_mean(self, value: float) -> float:
        t = torch.tensor(float(value), device=self.device)
        if self.distributed and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= self.world_size
        return float(t.item())

    def _build_loader(self, train_ds):
        num_workers = int(self.config["data"].get("num_workers", 0))
        sampler = None
        shuffle = True
        if self.distributed:
            sampler = DistributedSampler(
                train_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=False
            )
            shuffle = False
        loader = DataLoader(
            train_ds,
            batch_size=self.args.bs_train,
            drop_last=True,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        return loader, sampler

    def _build_ip2p(self):
        # This training path backpropagates through the edit model.
        # Keeping the frozen diffusion modules in fp32 is slower, but avoids
        # dtype flips / VAE upcasting issues that destabilize gradients in fp16.
        dtype = torch.float32
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.args.ip2p_model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        pipe.unet.eval()
        pipe.vae.eval()
        pipe.text_encoder.eval()
        for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
            for p in module.parameters():
                p.requires_grad_(False)
        return pipe

    @staticmethod
    def _resolve_instruction(instruction: str) -> str:
        if not instruction:
            return instruction
        key = instruction.strip()
        if key in INSTRUCTION_MAP:
            return INSTRUCTION_MAP[key]
        normalized = key.lower().replace(" ", "_")
        return INSTRUCTION_MAP.get(normalized, instruction)

    def _pipe_call_with_grad(self, **kwargs):
        fn = self.pipe.__class__.__call__
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        return fn(self.pipe, **kwargs)

    def edit(self, img01: torch.Tensor, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device).manual_seed(seed)
        pipe_dtype = next(self.pipe.unet.parameters()).dtype
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        image = img01.unsqueeze(0) if img01.dim() == 3 else img01
        image = image.to(device=self.device, dtype=pipe_dtype)
        latents = self._pipe_call_with_grad(
            prompt=self.instruction,
            image=image,
            generator=g,
            num_inference_steps=self.args.ip2p_steps,
            guidance_scale=self.args.guidance_scale,
            image_guidance_scale=self.args.image_guidance_scale,
            output_type="latent",
        ).images
        latents = latents.to(dtype=vae_dtype)
        imgs = self.pipe.vae.decode(
            latents / self.pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        return (imgs / 2 + 0.5).clamp(0, 1)

    def _iter_p_edit(self, iter_idx: int) -> float:
        if iter_idx < self.args.warmup_no_edit_iters:
            return 0.0
        if self.args.edit_ramp_iters <= 0:
            return 1.0
        x = (iter_idx - self.args.warmup_no_edit_iters + 1) / float(self.args.edit_ramp_iters)
        return float(max(0.0, min(1.0, x)))

    @staticmethod
    def _compute_roc_metrics(scores_pos: np.ndarray, scores_neg: np.ndarray):
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

    @staticmethod
    def _safe_fid(folder_a: Path, folder_b: Path, device: str):
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

    @staticmethod
    def _safe_inception(folder: Path, device: str):
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
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
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

    @staticmethod
    def _safe_clip_similarity(folder_a: Path, folder_b: Path, device: str,
                              clip_model: str = "openai/clip-vit-base-patch32"):
        _tv = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())
        if _tv < (2, 6):
            return None, f"CLIP skipped: torch {torch.__version__} < 2.6 required"
        try:
            from transformers import CLIPModel, CLIPProcessor
        except Exception as e:
            return None, str(e)
        try:
            model = CLIPModel.from_pretrained(clip_model).to(device).eval()
            processor = CLIPProcessor.from_pretrained(clip_model)
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

    def _evaluate_saved_outputs(self, uid_by_filename: Dict[str, int]):
        tfm = transforms.Compose([
            transforms.Resize((self.args.resolution, self.args.resolution)),
            transforms.ToTensor(),
        ])
        detector = self._unwrap_module(self.detector).eval()
        det_dtype = next(detector.parameters()).dtype

        split_scores = {"orig": [], "pre": [], "wm": []}
        split_correct = {"orig": 0.0, "pre": 0.0, "wm": 0.0}
        split_total = {"orig": 0, "pre": 0, "wm": 0}

        with torch.no_grad():
            for split in ["orig", "pre", "wm"]:
                split_dir = self.image_out_root / split
                for img_path in sorted(split_dir.glob("*.png")):
                    fn = img_path.name
                    if fn not in uid_by_filename:
                        continue
                    uid = int(uid_by_filename[fn])
                    img = Image.open(img_path).convert("RGB")
                    x = tfm(img).unsqueeze(0).to(self.device) * 2 - 1
                    if x.dtype != det_dtype:
                        x = x.to(det_dtype)
                    _, logits = detector(x)
                    probs = F.softmax(logits, dim=1)
                    pred = int(torch.argmax(logits, dim=1).item())
                    conf = float(torch.max(probs, dim=1).values.item())

                    split_scores[split].append(conf)
                    split_correct[split] += float(pred == uid)
                    split_total[split] += 1

        id_acc = {}
        for split in ["orig", "pre", "wm"]:
            denom = max(1, split_total[split])
            id_acc[split] = float(split_correct[split] / denom)

        auc_pre, acc_pre, tpr1_pre = self._compute_roc_metrics(
            np.asarray(split_scores["wm"], dtype=float),
            np.asarray(split_scores["pre"], dtype=float),
        )
        auc_orig, acc_orig, tpr1_orig = self._compute_roc_metrics(
            np.asarray(split_scores["wm"], dtype=float),
            np.asarray(split_scores["orig"], dtype=float),
        )

        quality = {"wm_vs_pre": {}, "wm_vs_orig": {}, "is_wm": {}}
        warnings = []
        device_str = str(self.device)
        fid_pre, err = self._safe_fid(self.image_out_root / "pre", self.image_out_root / "wm", device_str)
        quality["wm_vs_pre"]["fid"] = fid_pre
        if err:
            warnings.append(f"FID wm_vs_pre failed: {err}")
        fid_orig, err = self._safe_fid(self.image_out_root / "orig", self.image_out_root / "wm", device_str)
        quality["wm_vs_orig"]["fid"] = fid_orig
        if err:
            warnings.append(f"FID wm_vs_orig failed: {err}")

        is_mean, is_std, err = self._safe_inception(self.image_out_root / "wm", device_str)
        quality["is_wm"]["mean"] = is_mean
        quality["is_wm"]["std"] = is_std
        if err:
            warnings.append(f"IS wm failed: {err}")

        clip_pre, err = self._safe_clip_similarity(self.image_out_root / "pre", self.image_out_root / "wm", device_str)
        quality["wm_vs_pre"]["clip_sim"] = clip_pre
        if err:
            warnings.append(f"CLIP wm_vs_pre failed: {err}")
        clip_orig, err = self._safe_clip_similarity(self.image_out_root / "orig", self.image_out_root / "wm", device_str)
        quality["wm_vs_orig"]["clip_sim"] = clip_orig
        if err:
            warnings.append(f"CLIP wm_vs_orig failed: {err}")

        return {
            "id_acc": id_acc,
            "counts": {k: int(v) for k, v in split_total.items()},
            "verify_wm_vs_pre": {"auc": auc_pre, "acc": acc_pre, "tpr_at_1pct_fpr": tpr1_pre},
            "verify_wm_vs_orig": {"auc": auc_orig, "acc": acc_orig, "tpr_at_1pct_fpr": tpr1_orig},
            "quality": quality,
            "warnings": warnings,
        }

    def run(self):
        start = time.time()
        records = []

        global_batch = max(1, self.args.bs_train * self.world_size)
        max_batches_per_rank = max(1, self.args.n_train_img // global_batch)
        effective_n_train = max_batches_per_rank * global_batch
        filename_width = max(3, len(str(max(0, effective_n_train - 1))))
        if self.is_main and effective_n_train != self.args.n_train_img:
            print(
                f"[EmbedMark] n_train_img={self.args.n_train_img} adjusted to {effective_n_train} "
                f"for world_size={self.world_size}, bs_train={self.args.bs_train}"
            )

        for iter_idx in range(self.args.n_iter):
            if self.sampler is not None:
                self.sampler.set_epoch(iter_idx)
            p_edit = self._iter_p_edit(iter_idx)
            stats: Dict[str, float] = {
                "loss": 0.0,
                "id": 0.0,
                "sim": 0.0,
                "car": 0.0,
                "cons": 0.0,
                "acc_wm": 0.0,
                "acc_out": 0.0,
                "acc_mix": 0.0,
                "steps": 0.0,
            }

            pbar = None
            if self.is_main and int(getattr(self.args, "batch_pbar", 0)) == 1:
                pbar = tqdm(
                    total=max_batches_per_rank,
                    desc=f"iter {iter_idx+1}/{self.args.n_iter}",
                    leave=False,
                )

            for step, img in enumerate(self.loader):
                if step >= max_batches_per_rank:
                    break
                img = img.to(self.device)
                bsz = img.shape[0]
                img01 = ((img + 1) * 0.5).clamp(0, 1)

                batch_loss_total = 0.0
                batch_id_total = 0.0
                batch_sim_total = 0.0
                batch_car_total = 0.0
                batch_cons_total = 0.0
                batch_acc_wm_total = 0.0
                batch_acc_out_total = 0.0
                batch_acc_mix_total = 0.0

                sample_losses = []
                for b in range(bsz):
                    global_sample_idx = step * self.world_size * self.args.bs_train + self.rank * self.args.bs_train + b
                    global_id = iter_idx * self.args.n_train_img + global_sample_idx
                    uid = int(global_id % self.args.num_user)
                    user_id_t = torch.tensor([uid], device=self.device, dtype=torch.long)

                    img_b = img[b:b + 1]
                    img01_b = img01[b:b + 1]
                    wm = self.writer(img_b, user_id_t, self.args.wm_strength)
                    wm01 = ((wm + 1) * 0.5).clamp(0, 1)

                    seed = self.args.seed + int(global_id)
                    if p_edit > 0.0:
                        with torch.no_grad():
                            pre = self.edit(img01_b[0], seed)
                        out = self.edit(wm01[0], seed)
                    else:
                        pre = img01_b
                        out = wm01

                    out = out.to(self.device)
                    pre = pre.to(self.device)
                    if out.shape[-1] != self.args.resolution:
                        out = F.interpolate(out, size=(self.args.resolution, self.args.resolution), mode="bilinear", align_corners=False)
                    if pre.shape[-1] != self.args.resolution:
                        pre = F.interpolate(pre, size=(self.args.resolution, self.args.resolution), mode="bilinear", align_corners=False)

                    wm_t = wm
                    out_t = out * 2 - 1
                    det_dtype = next(self._unwrap_module(self.detector).parameters()).dtype
                    if wm_t.dtype != det_dtype:
                        wm_t = wm_t.to(det_dtype)
                    if out_t.dtype != det_dtype:
                        out_t = out_t.to(det_dtype)

                    emb_wm, logits_wm = self.detector(wm_t)
                    emb_out, logits_out = self.detector(out_t)

                    id_wm = F.cross_entropy(logits_wm, user_id_t)
                    id_out = F.cross_entropy(logits_out, user_id_t)
                    id_loss = self.args.wm_decode_lambda * id_wm + self.args.out_decode_lambda * p_edit * id_out

                    sim_loss = F.l1_loss(out, pre)
                    car_loss = F.l1_loss(wm01, img01_b)
                    cos = F.cosine_similarity(emb_wm, emb_out, dim=-1)
                    cons_loss = (1.0 - cos).mean()

                    loss = (
                        self.args.id_lambda * id_loss
                        + self.args.sim_lambda * p_edit * sim_loss
                        + self.args.carr_lambda * car_loss
                        + self.args.cons_lambda * p_edit * cons_loss
                    )
                    sample_losses.append(loss)

                    with torch.no_grad():
                        acc_wm = (torch.argmax(logits_wm, dim=1) == user_id_t).float().mean().item()
                        acc_out = (torch.argmax(logits_out, dim=1) == user_id_t).float().mean().item()
                        acc_mix = (1.0 - p_edit) * acc_wm + p_edit * acc_out

                    batch_loss_total += float(loss.item())
                    batch_id_total += float(id_loss.item())
                    batch_sim_total += float((p_edit * sim_loss).item())
                    batch_car_total += float(car_loss.item())
                    batch_cons_total += float((p_edit * cons_loss).item())
                    batch_acc_wm_total += float(acc_wm)
                    batch_acc_out_total += float(acc_out)
                    batch_acc_mix_total += float(acc_mix)

                    fn = f"{int(global_sample_idx):0{filename_width}d}.png"
                    is_saved = bool(iter_idx == self.args.n_iter - 1 and self.args.save_images)
                    if is_saved:
                        to_pil(img01_b[0]).save(self.image_out_root / "orig" / fn)
                        to_pil(pre[0]).save(self.image_out_root / "pre" / fn)
                        to_pil(out[0]).save(self.image_out_root / "wm" / fn)

                    records.append(
                        {
                            "index": int(global_id),
                            "uid": uid,
                            "seed": int(seed),
                            "filename": fn,
                            "saved": is_saved,
                            "instruction": self.instruction,
                            "p_edit": float(p_edit),
                        }
                    )

                batch_loss = torch.stack(sample_losses, dim=0).mean()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                local_loss = batch_loss_total / max(1, bsz)
                local_id = batch_id_total / max(1, bsz)
                local_sim = batch_sim_total / max(1, bsz)
                local_car = batch_car_total / max(1, bsz)
                local_cons = batch_cons_total / max(1, bsz)
                local_acc_wm = batch_acc_wm_total / max(1, bsz)
                local_acc_out = batch_acc_out_total / max(1, bsz)
                local_acc_mix = batch_acc_mix_total / max(1, bsz)

                disp_loss = self._reduce_mean(local_loss)
                disp_id = self._reduce_mean(local_id)
                disp_sim = self._reduce_mean(local_sim)
                disp_car = self._reduce_mean(local_car)
                disp_cons = self._reduce_mean(local_cons)
                disp_acc_wm = self._reduce_mean(local_acc_wm)
                disp_acc_out = self._reduce_mean(local_acc_out)
                disp_acc_mix = self._reduce_mean(local_acc_mix)

                stats["loss"] += disp_loss
                stats["id"] += disp_id
                stats["sim"] += disp_sim
                stats["car"] += disp_car
                stats["cons"] += disp_cons
                stats["acc_wm"] += disp_acc_wm
                stats["acc_out"] += disp_acc_out
                stats["acc_mix"] += disp_acc_mix
                stats["steps"] += 1.0

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        pedit=f"{p_edit:.2f}",
                        loss=f"{disp_loss:.4f}",
                        id=f"{disp_id:.4f}",
                        sim=f"{disp_sim:.4f}",
                        carr=f"{disp_car:.4f}",
                        cons=f"{disp_cons:.4f}",
                        awm=f"{disp_acc_wm:.3f}",
                        aout=f"{disp_acc_out:.3f}",
                    )
            if pbar is not None:
                pbar.close()

            self.scheduler.step()

            denom = max(1.0, stats["steps"])
            log_interval = max(1, int(getattr(self.args, "log_interval", 10)))
            should_log = ((iter_idx + 1) % log_interval == 0) or (iter_idx + 1 == self.args.n_iter)
            if self.is_main and should_log:
                print(
                    f"[EmbedMark][iter {iter_idx+1}/{self.args.n_iter}] "
                    f"p_edit={p_edit:.3f} "
                    f"loss={stats['loss']/denom:.6f} "
                    f"id={stats['id']/denom:.6f} sim={stats['sim']/denom:.6f} "
                    f"carr={stats['car']/denom:.6f} cons={stats['cons']/denom:.6f} "
                    f"acc_wm={stats['acc_wm']/denom:.4f} acc_out={stats['acc_out']/denom:.4f} "
                    f"acc_mix={stats['acc_mix']/denom:.4f}"
                )

        if self.distributed and dist.is_initialized():
            gathered_records = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_records, records)
            records_all = []
            for r in gathered_records:
                records_all.extend(r)
        else:
            records_all = records
        records_all = sorted(records_all, key=lambda x: x["index"])

        writer_state = self._unwrap_module(self.writer).state_dict()
        detector_state = self._unwrap_module(self.detector).state_dict()
        ckpt = {
            "writer": writer_state,
            "detector": detector_state,
            "args": vars(self.args),
            "instruction": self.instruction,
            "records": records_all,
        }
        if self.is_main:
            uid_by_filename = {}
            for r in records_all:
                fn = r.get("filename")
                uid = r.get("uid")
                saved = bool(r.get("saved", False))
                if (fn is None) or (uid is None) or (not saved):
                    continue
                uid_by_filename[str(fn)] = int(uid)

            eval_summary = None
            if int(getattr(self.args, "auto_eval", 1)) == 1 and int(self.args.save_images) == 1:
                eval_summary = self._evaluate_saved_outputs(uid_by_filename)

            torch.save(ckpt, self.out_root / "checkpoint.pt")

            summary = {
                "out_root": str(self.out_root),
                "image_out_root": str(self.image_out_root),
                "run_name": getattr(self.args, "run_name", self.out_root.name),
                "instruction": self.instruction,
                "num_user": self.args.num_user,
                "embed_dim": self.args.embed_dim,
                "args": vars(self.args),
                "records": records_all,
            }
            if eval_summary is not None:
                summary["auto_eval"] = eval_summary
            with open(self.out_root / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            if eval_summary is not None:
                with open(self.out_root / "eval_auto.json", "w", encoding="utf-8") as f:
                    json.dump(eval_summary, f, ensure_ascii=False, indent=2)

        elapsed = int(time.time() - start)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        if self.is_main:
            print(f"[EmbedMark] Saved to: {self.out_root}")
            print(f"[EmbedMark] Images saved to: {self.image_out_root}")
            if int(getattr(self.args, "auto_eval", 1)) == 1 and int(self.args.save_images) == 1:
                with open(self.out_root / "eval_auto.json", "r", encoding="utf-8") as f:
                    eval_summary = json.load(f)
                print(
                    "[EmbedMark][auto_eval] "
                    f"id_acc(orig/pre/wm)="
                    f"{eval_summary['id_acc']['orig']:.4f}/"
                    f"{eval_summary['id_acc']['pre']:.4f}/"
                    f"{eval_summary['id_acc']['wm']:.4f} "
                    f"wm_vs_pre(auc/acc/tpr1)="
                    f"{eval_summary['verify_wm_vs_pre']['auc']:.4f}/"
                    f"{eval_summary['verify_wm_vs_pre']['acc']:.4f}/"
                    f"{eval_summary['verify_wm_vs_pre']['tpr_at_1pct_fpr']:.4f} "
                    f"wm_vs_orig(auc/acc/tpr1)="
                    f"{eval_summary['verify_wm_vs_orig']['auc']:.4f}/"
                    f"{eval_summary['verify_wm_vs_orig']['acc']:.4f}/"
                    f"{eval_summary['verify_wm_vs_orig']['tpr_at_1pct_fpr']:.4f}"
                )
                q = eval_summary.get("quality", {})
                wm_pre = q.get("wm_vs_pre", {})
                wm_orig = q.get("wm_vs_orig", {})
                is_wm = q.get("is_wm", {})
                print(
                    "[EmbedMark][auto_eval_quality] "
                    f"fid(wm,pre)={wm_pre.get('fid')} "
                    f"fid(wm,orig)={wm_orig.get('fid')} "
                    f"is(wm)={is_wm.get('mean')}±{is_wm.get('std')} "
                    f"clip(wm,pre)={wm_pre.get('clip_sim')} "
                    f"clip(wm,orig)={wm_orig.get('clip_sim')}"
                )
                for w in eval_summary.get("warnings", []):
                    print(f"[EmbedMark][auto_eval_warning] {w}")
            print(f"[EmbedMark] Training time: {h:02d}:{m:02d}:{s:02d}")
