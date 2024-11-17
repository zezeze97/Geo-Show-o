export PYTHONPATH=/lustre/home/2001110054/Show-o:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python reconstruct_geo.py --config_file /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1115_mask_z13/show/config.yaml \
                                                --ckpt_path /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1115_mask_z13/ckpt/epoch=245-step=91512.ckpt \
                                                --image_size 512 \
                                                --batch_size 1 \
                                                --output_dir /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1115_mask_z13/formalgeo_reconstruct \
                                                --data_path /lustre/home/2001110054/GEO-Open-MAGVIT2/geo_data/val