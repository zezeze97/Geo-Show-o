python3 inference_mmu.py config=configs/geouni_test_512x512.yaml \
max_new_tokens=384 \
mmu_image_root=./data/formalgeo7k/formalgeo7k_v2 \
validation_prompts_file=data/formalgeo7k/formalgeo7k_v2/custom_json/qa_structure_only/qa_structure_only_minitrain.json \
output_dir='outputs/geouni-512x512-0130-t2i-mmu-debug/formalization' \
save_file_name='formalization'
