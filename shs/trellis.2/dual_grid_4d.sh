python data_toolkit/dual_grid_v2.py \
  --ann_file data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
  --rendered_root data/objverse_minghao_4d_mine_40075/rendering_v5 \
  --output_root data/trellis.2/dual_grid_4d \
  --resolution 512 \
  --max_workers 8 \
  --world_size ${1:-1} --rank ${2:-0}
