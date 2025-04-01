export PYTHONPATH=/lustre/home/2001110054/Geo-Show-o:$PYTHONPATH

CONFIG=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/ablation_normal/test/config.yaml
CKPT=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/ablation_normal/epoch=48-step=18228.ckpt
SAVE_PATH=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/ablation_normal

CUDA_VISIBLE_DEVICES=0 python geo_utils/test_geo_vq_model.py --config $CONFIG --ckpt $CKPT --save_path $SAVE_PATH
CUDA_VISIBLE_DEVICES=0 python geo_utils/draw_vq_distribution.py --config $CONFIG --ckpt $CKPT --save_path $SAVE_PATH


