# SwiftEdit EmbedMark

This directory keeps the current EmbedMark training setup, but uses a lightweight text-guided turbo image editor plus a soft spatial blend step to preserve more of the original background.

Current implementation:
- shared dataset/config/output logic with `../instructPix2Pix`
- editor backend: `diffusers.StableDiffusionImg2ImgPipeline`
- default editor checkpoint: `stabilityai/sd-turbo`
- optional LoRA adapter support via `--swift_lora_path`
- differentiable soft mask blend controlled by `--swift_mask_threshold`, `--swift_mask_sharpness`, and `--swift_blend`

Notes:
- this is a local SwiftEdit-style backend for your benchmark codebase, not a bundled official upstream repo
- if you later add a local checkpoint or adapter, pass it with `--swift_model_id`
- outputs remain compatible with the same `summary.py` style aggregation

Example:
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 main.py --config ../instructPix2Pix/configs/celeba.yml --num_user 100 --fp16 1 --bs_train 2 --swift_steps 1 --swift_strength 0.6 --swift_mask_threshold 0.04 --swift_mask_sharpness 30 --swift_blend 1.0 --embed_dim 128 --user_dim 64 --writer_hidden 96 --writer_blocks 4 --wm_strength 0.07 --carr_lambda 0.85 --sim_lambda 2.0 --id_lambda 3.2 --out_decode_lambda 1.4 --cons_lambda 0.8 --n_train_img 1404 --n_iter 170 --lr 1.8e-4 --warmup_no_edit_iters 35 --edit_ramp_iters 90 --save_images 1 --auto_eval 1 --clean_image_out 1 --batch_pbar 0 --seed 1234 --instruction beards1
```
