python data_toolkit/download_by_sha256.py \
  --sha256_list claude_tmp/uuid_github_intersection_sha256.txt \
  --root trellis.2_data/ObjaverseXL_github \
  --download_root trellis.2_data/dynamic_obj \
  --world_size ${1:-1} --rank ${2:-0}
