# IP2P EmbedMark (New Architecture)

This is a standalone implementation that keeps the editor as InstructPix2Pix, but replaces the old bit-decoder setup with user-embedding retrieval.

## Core idea

- `writer`: writes a user-conditioned residual watermark on input image.
- `editor`: InstructPix2Pix edits the clean and watermarked images.
- `detector`: predicts user identity from normalized embedding + cosine prototypes.

Training is staged:

- warmup: no edit (`p_edit = 0`) to learn watermark identity path first.
- ramp: `p_edit` increases linearly so edit-robust identity is learned progressively.

## Files

- `main.py`: CLI entry.
- `train.py`: training loop and IP2P calls.
- `models.py`: writer and detector modules.

## Run

```bash
/home/nvidia/miniconda3/envs/ip2p/bin/python /home/nvidia/Sheldon/tracemark/instructPix2Pix/main.py \
  --config /home/nvidia/Sheldon/tracemark/instructPix2Pix/configs/celeba.yml \
  --instruction beards1 \
  --num_user 15 \
  --fp16 1 \
  --warmup_no_edit_iters 10 \
  --edit_ramp_iters 20
```

Multi-GPU (DDP):

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/nvidia/miniconda3/envs/ip2p/bin/torchrun \
  --nproc_per_node=4 /home/nvidia/Sheldon/tracemark/instructPix2Pix/main.py \
  --config /home/nvidia/Sheldon/tracemark/instructPix2Pix/configs/celeba.yml \
  --instruction beards1 \
  --num_user 15 \
  --fp16 1 \
  --warmup_no_edit_iters 10 \
  --edit_ramp_iters 20
```

Recommended 4-GPU stable starting point for `beards1` with `num_user=100`:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/nvidia/miniconda3/envs/ip2p/bin/torchrun \
  --nproc_per_node=4 /home/nvidia/Sheldon/tracemark/instructPix2Pix/main.py \
  --config /home/nvidia/Sheldon/tracemark/instructPix2Pix/configs/celeba.yml \
  --instruction beards1 \
  --num_user 100 \
  --fp16 1 \
  --bs_train 2 \
  --embed_dim 128 \
  --user_dim 64 \
  --writer_hidden 96 \
  --writer_blocks 4 \
  --wm_strength 0.075 \
  --carr_lambda 0.85 \
  --sim_lambda 1.6 \
  --id_lambda 3.2 \
  --out_decode_lambda 1.4 \
  --cons_lambda 0.8 \
  --n_train_img 1400 \
  --n_iter 170 \
  --lr 1.8e-4 \
  --warmup_no_edit_iters 35 \
  --edit_ramp_iters 90 \
  --save_images 1 \
  --auto_eval 1 \
  --clean_image_out 1 \
  --batch_pbar 0 \
  --seed 1234
```

`bs_train` is set to `2` here so the effective global batch stays at `8` on 4 GPUs, matching the old 2-GPU run with `bs_train=4`.

Outputs are written to:

- `out/<instruction>/<run_name>/train.log`
- `out/<instruction>/<run_name>/args.json`
- `out/<instruction>/<run_name>/checkpoint.pt`
- `out/<instruction>/<run_name>/summary.json`
- `out/<instruction>/<run_name>/eval_auto.json` (when `--auto_eval 1`, includes id/verify + FID/IS/CLIP quality)
- `out/<instruction>/<run_name>/{orig,pre,wm}` (when `--save_images 1`)
