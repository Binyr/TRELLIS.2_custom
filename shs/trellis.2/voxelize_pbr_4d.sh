DATA_ROOT=/threed-code/yanruibin/efs/4D_video_data_process/data

python data_toolkit/voxelize_pbr_v2.py \
  --ann_file $DATA_ROOT/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
  --pbr_shared_root $DATA_ROOT/trellis.2/pbr_shared \
  --rendered_root $DATA_ROOT/objverse_minghao_4d_mine_40075/rendering_v5 \
  --output_root $DATA_ROOT/trellis.2/pbr_voxels_4d \
  --resolution 1024 \
  --max_workers 1 \
  --priority_list claude_tmp/objv1_sketchfab_intersection.txt \
  --finished_views $DATA_ROOT/trellis.2/pbr_voxels_4d/finished_views_512.json \
  --world_size ${1:-1} --rank ${2:-0}
