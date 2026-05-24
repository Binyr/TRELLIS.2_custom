#!/usr/bin/env bash
# Resubmit pbr ranks: not running + last status avg > 2000s/view (2026-05-23)

num_chunk=170

for i in 9 10 11 12 13 14 15 16 18 20 21 22 26 27 28 30 31 35 44 48 54 56 60 61 62 63 64 65 66 67 68 69 70 76 78 85 86 89 91 92 93 97 98 99 105 107 108 109 112 115 118 119 120 121 123 125 128 131 133 137 138 139 142 143 145 146 157 162 167; do
    echo "Submitting job for chunk $i, num_chunk $num_chunk"
    koala submit -m normal \
        --code s3://arcwm-code-us-west-2/yanruibin/code_1779550875/run_codes:/data/work/run_codes \
        -g 0 --cpu 12 --mem 48 \
        --image 600627331169.dkr.ecr.us-west-2.amazonaws.com/arcwm/train-aws:cuda12.8-efa1.44-ubuntu24.04-uvcache \
        -c "cd /data/work/run_codes && ls -lh && nvidia-smi && source uv/setup.sh --new-env --venv-dir /local-ssd/trellis.2-venv && bash shs/trellis.2/voxelize_pbr_4d.sh $num_chunk $i" \
        -j "pbr-$i"
    sleep 5
done
