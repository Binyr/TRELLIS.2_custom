koala submit -m normal --sync-code .:/data/work/run_codes -g 0 \
    --image 600627331169.dkr.ecr.us-west-2.amazonaws.com/arcwm/train-aws:cuda12.8-efa1.44-ubuntu24.04-uvcache \
    -c "cd /data/work/run_codes && ls -lh && source uv/setup.sh --new-env --venv-dir /local-ssd/vpixal3d-venv && export HF_TOKEN=\$HF_TOKEN && bash shs/download_sketch.sh" \
    -j "download-sketch"