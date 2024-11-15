export PYTHONPATH=/lustre/home/2001110054/Show-o:$PYTHONPATH

CONFIG=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1114_mask_z13/show/config.yaml
CKPT=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1114_mask_z13/ckpt/epoch=198-step=74028.ckpt

python draw_vq_distribution.py --config $CONFIG --ckpt $CKPT

python test_geo_vq_model.py --config $CONFIG --ckpt $CKPT
