#!/usr/bin/env bash
# Resubmit pbr ranks: not running + last status avg > 1000s/view (2026-05-27)

num_chunk=170

for i in 0 3 6 12 17 20 23 24 33 35 43 50 51 54 62 63 66 70 80 86 108 109 113 119 120 122 123 124 125 126 131 132 133 134 135 136 137 138 139 140 141 143 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169; do
    echo "Submitting job for chunk $i, num_chunk $num_chunk"
    koala submit -m normal \
        --code s3://arcwm-code-us-west-2/yanruibin/code_1779550875/run_codes:/data/work/run_codes \
        -g 0 --cpu 12 --mem 48 \
        --image 600627331169.dkr.ecr.us-west-2.amazonaws.com/arcwm/train-aws:cuda12.8-efa1.44-ubuntu24.04-uvcache \
        -c "cd /data/work/run_codes && ls -lh && nvidia-smi && source uv/setup.sh --new-env --venv-dir /local-ssd/trellis.2-venv && bash shs/trellis.2/voxelize_pbr_4d.sh $num_chunk $i" \
        -j "pbr-$i"
    sleep 5
done
