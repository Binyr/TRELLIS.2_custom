python data_toolkit/voxelize_pbr_v2.py \
  --ann_file data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
  --pbr_shared_root data/trellis.2/pbr_shared \
  --rendered_root data/objverse_minghao_4d_mine_40075/rendering_v5 \
  --output_root data/trellis.2/pbr_voxels_4d \
  --log_root data/trellis.2/logs/voxelize_pbr_4d \
  --resolution 512 \
  --max_workers 8 \
  --priority_list claude_tmp/objv1_sketchfab_intersection.txt \
  --world_size ${1:-1} --rank ${2:-0}
