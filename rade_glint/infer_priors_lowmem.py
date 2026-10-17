"""Run pinned official entrypoint with documented memory-only adaptations.

Decode chunk=1 changes temporal decoder context; this is recorded, not claimed
bit-identical to the official default. No test image or GLINT teacher is used.
"""
from pathlib import Path
import os,sys,hashlib,json
root=Path(__file__).resolve().parent
vendor=root/'vendor/diffusion-renderer'
path=vendor/'inference_svd_rgbx.py'
source=path.read_text()
old='    pipeline = pipeline.to(distributed_state.device)\n    pipeline = pipeline.to(weight_dtype)'
new='''    pipeline = pipeline.to(weight_dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
    # Independent 2D VAE image encodes, preserving mode() and input order.
    original_encode = pipeline._encode_vae_image
    def bounded_encode(image, device, num_videos_per_prompt, do_classifier_free_guidance):
        if do_classifier_free_guidance:
            return original_encode(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        return torch.cat([original_encode(part, device, num_videos_per_prompt, False)
                          for part in image.split(1)], dim=0)
    pipeline._encode_vae_image = bounded_encode'''
assert source.count(old)==1
source=source.replace(old,new)
(root/'prior_runtime.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    cpu_offload=True,unet_chunk_size=1,vae_encode_chunk=1,decode_chunk_size=1,
    resolution=[512,384],frames=24,steps=20,cam00_used=False),indent=2))
sys.path.insert(0,str(vendor));os.chdir(vendor)
sys.argv=[str(path),'--config','configs/rgbx_inference.yaml',
    f'inference_input_dir={root}/prior_inputs',f'inference_save_dir={root}/prior_outputs',
    'inference_res=[384,512]','inference_n_frames=24','inference_n_steps=20',
    'model_passes=[normal,depth,diffuse_albedo,basecolor]','seed=0',
    'save_video=false','save_image=true','decode_chunk_size=1']
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
