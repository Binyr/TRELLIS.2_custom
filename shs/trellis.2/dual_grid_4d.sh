python data_toolkit/dual_grid_v2.py \
  --ann_file data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
  --rendered_root data/objverse_minghao_4d_mine_40075/rendering_v5 \
  --output_root /threed-code/yanruibin/efs/4D_video_data_process/data/trellis.2/dual_grid_4d_v2 \
  --resolution 512 \
  --max_workers 8 \
  --priority_list claude_tmp/objv1_sketchfab_intersection.txt \
  --finished_views /local-ssd/data2/trellis.2/dual_grid_4d_v3/finished_views_512.json \
  --world_size ${1:-1} --rank ${2:-0}
