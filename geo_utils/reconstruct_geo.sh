export PYTHONPATH=/lustre/home/2001110054/Geo-Show-o:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python geo_utils/reconstruct_geo.py --config_file /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/ablation_normal/test/config.yaml \
                                                --ckpt_path /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/ablation_normal/epoch=48-step=18228.ckpt \
                                                --image_size 512 \
                                                --batch_size 1 \
                                                --output_dir /lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/ablation_normal/formalgeo_reconstruct \
                                                --data_path /lustre/home/2001110054/GEO-Open-MAGVIT2/geo_data/formalgeo/val