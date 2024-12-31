python3 geo_inference_t2i.py config=configs/geo_showo_demo_512x512.yaml \
batch_size=1 validation_prompts_file=validation_prompts/t2i_merged.txt \
guidance_scale=5 generation_timesteps=100 \
mode='t2i'
