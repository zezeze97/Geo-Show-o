export PYTHONPATH=/lustre/home/2001110054/Geo-Show-o:$PYTHONPATH

CONFIG=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_0312_mask_down32_z13/32down/config.yaml
CKPT=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_0312_mask_down32_z13/ckpt/epoch=220-step=147344.ckpt
SAVE_PATH=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_0312_mask_down32_z13

CUDA_VISIBLE_DEVICES=0 python geo_utils/test_geo_vq_model.py --config $CONFIG --ckpt $CKPT --save_path $SAVE_PATH
CUDA_VISIBLE_DEVICES=0 python geo_utils/draw_vq_distribution.py --config $CONFIG --ckpt $CKPT --save_path $SAVE_PATH


