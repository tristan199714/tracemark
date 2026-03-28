# pix2pix-turbo EmbedMark

This directory reuses the current EmbedMark training pipeline, datasets, and evaluation format from `../instructPix2Pix`, but swaps the editor backend to a one-step turbo image-to-image pipeline.

Current implementation:
- shared dataset/config/output logic with `../instructPix2Pix`
- editor backend: `diffusers.StableDiffusionImg2ImgPipeline`
- default editor checkpoint: `stabilityai/sd-turbo`
- optional LoRA adapter support via `--turbo_lora_path`

Notes:
- this is a diffusers-compatible turbo backend for your benchmark codebase
- if you later add an official pix2pix-turbo checkpoint or local path, pass it with `--turbo_model_id`
- outputs remain compatible with the same `summary.py` style aggregation

Example:
```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 main.py --config ../instructPix2Pix/configs/celeba.yml --num_user 100 --fp16 1 --bs_train 2 --turbo_steps 1 --turbo_strength 0.55 --embed_dim 128 --user_dim 64 --writer_hidden 96 --writer_blocks 4 --wm_strength 0.07 --carr_lambda 0.85 --sim_lambda 2.0 --id_lambda 3.2 --out_decode_lambda 1.4 --cons_lambda 0.8 --n_train_img 1404 --n_iter 170 --lr 1.8e-4 --warmup_no_edit_iters 35 --edit_ramp_iters 90 --save_images 1 --auto_eval 1 --clean_image_out 1 --batch_pbar 0 --seed 1234 --instruction beards1
```
