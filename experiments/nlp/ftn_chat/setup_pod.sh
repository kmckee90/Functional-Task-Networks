#!/usr/bin/env bash
# One-time pod setup after `actl pod up` / pod recreation (run ON the pod).
# Qwen3.6's Gated DeltaNet fast path needs, beyond the Python deps:
#   - gcc + python3 dev headers: triton JIT-compiles kernels at runtime.
#   - cuda-nvcc-12-6: fla's gated-DeltaNet backward is wrong on Hopper with
#     triton>=3.4 (fla #640) and reroutes through tilelang, which needs nvcc.
#     The runtime toolkit here is 12.6; nvcc 12.6 compiling Hopper (sm_90)
#     kernels runs fine against the newer driver. Export CUDA_HOME so tilelang
#     finds it (train.py / qa_synth.py / bench_speed.py read the env).
set -e
apt-get update -qq
apt-get install -y -qq gcc python3-dev libpython3.12-dev cuda-nvcc-12-6
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
echo "pod setup done: $(gcc --version | head -1); nvcc: $(/usr/local/cuda-12.6/bin/nvcc --version | tail -1)"
