num_chunk=16
for i in $(seq 0 $((num_chunk-1))); do
    echo "Submitting job for chunk $i, num_chunk $num_chunk"
    koala submit -m normal --sync-code .:/data/work/run_codes -g 0 \
    --image 600627331169.dkr.ecr.us-west-2.amazonaws.com/arcwm/train-aws:cuda12.8-efa1.44-ubuntu24.04-uvcache \
    -c "cd /data/work/run_codes && git config --global credential.helper store && git config --global url.\"https://\$GH_TOKEN@github.com/\".insteadOf \"https://github.com/\" && source uv/setup.sh --new-env --venv-dir /local-ssd/vpixal3d-venv && export HF_TOKEN=\$HF_TOKEN && bash shs/download_dynamic_obj.sh $num_chunk $i" \
    -j "download-dynamic-obj-$i"
done
