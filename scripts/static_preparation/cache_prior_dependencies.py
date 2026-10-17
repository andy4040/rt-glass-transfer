from huggingface_hub import snapshot_download

# These are the official inference entrypoint's two additional components.
print(snapshot_download('stabilityai/stable-video-diffusion-img2vid',
    revision='9cf024d5bfa8f56622af86c884f26a52f6676f2e',
    allow_patterns=['image_encoder/config.json','image_encoder/model.safetensors','feature_extractor/*']))
