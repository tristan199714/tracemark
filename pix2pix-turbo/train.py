import importlib.util
import sys
from pathlib import Path

import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline

THIS_DIR = Path(__file__).resolve().parent
SHARED_ROOT = THIS_DIR.parent / "instructPix2Pix"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))


def _load_shared_train_module():
    spec = importlib.util.spec_from_file_location("shared_instruct_train", SHARED_ROOT / "train.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


IP2PEmbedMarkTrainer = _load_shared_train_module().IP2PEmbedMarkTrainer


class Pix2PixTurboEmbedMarkTrainer(IP2PEmbedMarkTrainer):
    def _build_ip2p(self):
        dtype = torch.float16 if self.args.fp16 else torch.float32
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.args.turbo_model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
        if getattr(self.args, "turbo_lora_path", ""):
            pipe.load_lora_weights(self.args.turbo_lora_path)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
            if module is None:
                continue
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
        return pipe

    @staticmethod
    def _normalize_pipe_output(images: torch.Tensor) -> torch.Tensor:
        if isinstance(images, (list, tuple)):
            images = images[0]
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[1] != 3 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        return images.clamp(0, 1)

    def edit(self, img01: torch.Tensor, seed: int) -> torch.Tensor:
        g = torch.Generator(device=self.device).manual_seed(seed)
        image = img01.unsqueeze(0) if img01.dim() == 3 else img01
        image = image.to(self.device)
        outputs = self._pipe_call_with_grad(
            prompt=self.instruction,
            negative_prompt=getattr(self.args, "negative_prompt", "") or None,
            image=image,
            generator=g,
            strength=self.args.turbo_strength,
            num_inference_steps=self.args.turbo_steps,
            guidance_scale=self.args.guidance_scale,
            output_type="pt",
        )
        return self._normalize_pipe_output(outputs.images)
