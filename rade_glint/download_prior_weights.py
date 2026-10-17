from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parent
snapshot_download('nexuslrf/diffusion_renderer-inverse-svd',
    revision='54243c27ad6299e43159410188403f7136b2dcba',max_workers=2,
    local_dir=str(root/'vendor/diffusion-renderer/checkpoints/diffusion_renderer-inverse-svd'))
snapshot_download('stabilityai/stable-video-diffusion-img2vid',
    revision='9cf024d5bfa8f56622af86c884f26a52f6676f2e',max_workers=2,
    allow_patterns=['image_encoder/config.json','image_encoder/model.safetensors','feature_extractor/*'])
