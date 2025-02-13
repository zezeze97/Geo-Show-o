export PYTHONPATH=/lustre/home/2001110054/Geo-Show-o:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python reconstruct_geo.py --config_file /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1120_mask_down32_z13/32down/config.yaml \
                                                --ckpt_path /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1120_mask_down32_z13/ckpt/epoch=184-step=68820.ckpt \
                                                --image_size 512 \
                                                --batch_size 1 \
                                                --output_dir /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1120_mask_down32_z13/formalgeo_reconstruct \
                                                --data_path /lustre/home/2001110054/GEO-Open-MAGVIT2/geo_data/val