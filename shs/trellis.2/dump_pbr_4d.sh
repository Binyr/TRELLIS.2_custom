python tools/run_dump_pbr_4d.py \
  --ann_file data/objverse_minghao_4d_mine_40075/rendering_v5_anns_8cam.json \
  --glb_root /threed-code/yanruibin/yanruibin/glbs \
  --output_root /threed-code/yanruibin/efs/4D_video_data_process/data/trellis.2/pbr_shared \
  --log_root /threed-code/yanruibin/efs/4D_video_data_process/data/trellis.2/pbr_shared/logs/ \
  --max_workers 8 \
  --priority_list claude_tmp/objv1_sketchfab_intersection.txt \
  --world_size ${1:-1} --rank ${2:-0}
