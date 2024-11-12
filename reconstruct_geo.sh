CUDA_VISIBLE_DEVICES=0 python reconstruct_geo.py --config_file /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1110_mask/show/config.yaml \
                                                --ckpt_path /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1110_mask/ckpt/epoch=135-step=50592.ckpt \
                                                --image_size 512 \
                                                --batch_size 1 \
                                                --output_dir /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1110_mask/formalgeo_reconstruct \
                                                --data_path /lustre/home/2001110054/GEO-Open-MAGVIT2/geo_data/val