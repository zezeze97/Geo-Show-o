export PYTHONPATH=/lustre/home/2001110054/Show-o:$PYTHONPATH

CONFIG=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1120_mask_z13_256/show/config.yaml
CKPT=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1120_mask_z13_256/ckpt/epoch=197-step=73656.ckpt
SAVE_PATH=/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1120_mask_z13_256/

python test_geo_vq_model.py --config $CONFIG --ckpt $CKPT --save_path $SAVE_PATH
python draw_vq_distribution.py --config $CONFIG --ckpt $CKPT --save_path $SAVE_PATH


