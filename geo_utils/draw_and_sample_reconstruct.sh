export PYTHONPATH=/lustre/home/2001110054/Show-o:$PYTHONPATH

CONFIG=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1115_mask_z13/show/config.yaml
CKPT=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1115_mask_z13/ckpt/epoch=245-step=91512.ckpt

python draw_vq_distribution.py --config $CONFIG --ckpt $CKPT

python test_geo_vq_model.py --config $CONFIG --ckpt $CKPT
