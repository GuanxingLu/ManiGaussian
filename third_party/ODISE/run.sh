# bash run.sh
export ODISE_MODEL_ZOO="/data/yanjieze/projects/nerf-act/archive"
CUDA_VISIBLE_DEVICES=7 python demo/demo.py --config-file configs/Panoptic/odise_caption_coco_50e.py \
  --input 12.png  \
  --init-from /data/yanjieze/projects/nerf-act/archive/odise_caption_coco_50e-853cc971.pth \
  --vocab "a robot arm is opening the drawer" \
  --output results/
  
