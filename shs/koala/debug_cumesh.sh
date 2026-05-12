koala submit -m normal --sync-code .:/data/work/run_codes -g 1 \
    --image 600627331169.dkr.ecr.us-west-2.amazonaws.com/arcwm/train-aws:cuda12.8-efa1.44-ubuntu24.04-uvcache \
    -c "cd /data/work/run_codes && source uv/setup.sh --new-env --venv-dir /local-ssd/trellis.2-venv 2>&1; echo '=== Now testing CuMesh ==='; source /local-ssd/trellis.2-venv/bin/activate && uv pip install git+https://github.com/JeffreyXiang/CuMesh.git --no-build-isolation -v 2>&1 | tail -100" \
    -j "debug-cumesh-build"
