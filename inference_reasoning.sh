CUDA_VISIBLE_DEVICES=0 python inference_reasoning.py config=configs/geouni_512x512_0221.yaml \
pretrained_geouni_model_path='checkpoints/geo-grpo-0228-4GPU' \
output_dir='outputs/geo-grpo-0228-4GPU' \
save_file_name='test_reasoning_choice_cn' \
mmu_image_root='/lustre/home/2001110054/Geo-Show-o/data/formalgeo7k/formalgeo7k_v2' \
validation_prompts_file='/lustre/home/2001110054/Geo-Show-o/data/formalgeo7k/formalgeo7k_v2/custom_json/geouni/reasoning/test_reasoning_choice_cn.jsonl' \
language='cn' \
formalization=False \
max_new_tokens=3000
