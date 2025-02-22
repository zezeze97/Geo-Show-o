CUDA_VISIBLE_DEVICES=0 python inference_reasoning.py config=configs/geouni_512x512_0221.yaml \
output_dir='temp_output' \
save_file_name='debug' \
mmu_image_root='/lustre/home/2001110054/Geo-Show-o/data/formalgeo7k/formalgeo7k_v2' \
validation_prompts_file='/lustre/home/2001110054/Geo-Show-o/data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/reasoning/test_reasoning_cn.jsonl' \
language='cn' \
formalization=False \
max_new_tokens=512
