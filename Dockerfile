# ==============================================================================
# 极速构建 + 算子预编译版 Dockerfile (专为 A800 / H20 / H200 / B300 定制)
# Base: NVIDIA CUDA 13.3.0 Devel + Ubuntu 24.04 (自带 nvcc 13.3)
# 预编译: DeepSpeed (fused_adam/cpu_adam/aio/utils) + FlashAttn + CausalConv1d + Megatron
# ==============================================================================
FROM nvcr.io/nvidia/cuda:13.3.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Hong_Kong \
    # 锁定 A800(8.0), H20/H200(9.0), B200/B300(10.0;10.3)
    TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;10.3" \
    CONDA_DIR=/opt/conda \
    PATH=/root/.local/bin:/root/.bun/bin:/opt/conda/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=300 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# ------------------------------------------------------------------------------
# 1. 单次 apt-get 安装系统工具、FFmpeg、libaio-dev (DeepSpeed AIO 必需) 与 RDMA 库
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    tmux vim nano htop nvtop iotop lsof tree fzf jq bat \
    ripgrep fd-find \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev libsndfile1-dev libsox-dev \
    git git-lfs curl wget unzip zip tar xz-utils zstd \
    net-tools iproute2 iputils-ping openssh-client openssh-server sudo ca-certificates gnupg \
    build-essential cmake ninja-build pkg-config \
    libnuma-dev libaio-dev libssl-dev libffi-dev libgl1 libglib2.0-0 \
    libibverbs-dev rdma-core ibverbs-utils && \
    git lfs install && \
    ln -sf /usr/bin/fdfind /usr/local/bin/fd && \
    ln -sf /usr/bin/batcat /usr/local/bin/bat && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 && \
    chmod +x /usr/local/bin/yq && \
    GH_VER=$((curl -fsSL https://api.github.com/repos/cli/cli/releases/latest || echo '{"tag_name":"v2.67.0"}') | jq -r '.tag_name' | tr -d 'v') && \
    GH_VER=${GH_VER:-2.67.0} && \
    curl -fsSL "https://github.com/cli/cli/releases/download/v${GH_VER}/gh_${GH_VER}_linux_amd64.tar.gz" | tar -xz -C /tmp && \
    mv /tmp/gh_${GH_VER}_linux_amd64/bin/gh /usr/local/bin/gh && rm -rf /tmp/gh_*

# ------------------------------------------------------------------------------
# 2. 极速安装 Conda (自带 Python 3.12 的 Miniforge3) + Node.js v22 二进制 + Bun + uv
# ------------------------------------------------------------------------------
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-24.11.3-0-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda init bash && \
    echo "conda activate base" >> /root/.bashrc && \
    curl -fsSL https://nodejs.org/dist/v22.19.0/node-v22.19.0-linux-x64.tar.xz | tar -xJ --strip-components=1 -C /usr/local && \
    curl -fsSL https://bun.sh/install | bash && \
    pip install --no-cache-dir uv

# ------------------------------------------------------------------------------
# 3. 安装 AI Coding Agent 套件: herdr + pi + BeaconBay/ck + qmd
# ------------------------------------------------------------------------------
RUN curl -fsSL https://herdr.dev/install.sh | sh && \
    npm install -g --ignore-scripts @earendil-works/pi-coding-agent && \
    npm install -g @tobilu/qmd @beaconbay/ck-search && \
    npm cache clean --force && \
    CK_URL=$((curl -fsSL https://api.github.com/repos/BeaconBay/ck/releases/latest || echo "{}") \
      | jq -r '.assets[]? | select(.name | contains("x86_64-unknown-linux-gnu.tar.gz")) | .browser_download_url' \
      | head -n 1) && \
    if [ -n "$CK_URL" ]; then \
      curl -fsSL "$CK_URL" | tar -xz -C /usr/local/bin && chmod +x /usr/local/bin/ck; \
    fi

# ------------------------------------------------------------------------------
# 4. 安装 vLLM 0.31.0 底座并锁定 PyTorch 版本
# ------------------------------------------------------------------------------
RUN uv pip install --no-cache "vllm==0.31.0" && \
    pip freeze | grep -E "^(torch|triton|torchvision|torchaudio)==" | sed 's/+.*//' > /tmp/torch_constraints.txt && \
    cat /tmp/torch_constraints.txt

# ------------------------------------------------------------------------------
# 5. 秒装预编译 flash-attn 与 causal-conv1d
# ------------------------------------------------------------------------------
RUN PY_TAG=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')") && \
    TORCH_MM=$(python -c "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))") && \
    CUDA_MAJ=$(python -c "import torch; print((torch.version.cuda or '13.0').split('.')[0])") && \
    FA_URL=$((curl -fsSL "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases?per_page=10" || echo "[]") \
      | jq -r --arg py "$PY_TAG" --arg tm "torch${TORCH_MM}" --arg cu "cu${CUDA_MAJ}" \
        '.[].assets[]?.browser_download_url | select(endswith("linux_x86_64.whl") and contains($py) and contains($tm) and contains($cu))' \
      | head -n 1) && \
    if [ -n "$FA_URL" ]; then uv pip install --no-cache --no-deps "$FA_URL"; fi && \
    CC_URL=$((curl -fsSL "https://api.github.com/repos/Dao-AILab/causal-conv1d/releases?per_page=5" || echo "[]") \
      | jq -r --arg py "$PY_TAG" --arg tm "torch${TORCH_MM}" --arg cu "cu${CUDA_MAJ}" \
        '.[].assets[]?.browser_download_url | select(endswith("linux_x86_64.whl") and contains($py) and contains($tm) and contains($cu))' \
      | head -n 1) && \
    if [ -n "$CC_URL" ]; then uv pip install --no-cache --no-deps "$CC_URL"; fi

# ------------------------------------------------------------------------------
# 6. 一次性并发安装 JupyterLab + ms-swift 4.6 + VLM + RL + 决策模型依赖
# ------------------------------------------------------------------------------
RUN uv pip install --no-cache \
    -c /tmp/torch_constraints.txt \
    nvitop ast-grep-cli pybind11 wheel packaging ninja psutil \
    jupyterlab notebook ipykernel ipywidgets \
    matplotlib pandas seaborn scipy scikit-learn \
    flash-linear-attention \
    "transformers>=5.10.4" \
    "modelscope>=1.23.0" \
    "datasets>=3.0,<4.8.5" \
    "peft>=0.11,<0.21" \
    "trl>=0.15,<1.0" \
    "deepspeed>=0.14,<0.19" \
    "accelerate" "optimum" "bitsandbytes" "mcore-bridge" "liger-kernel" \
    "gradio<5.33" "huggingface_hub[cli,hf_transfer]" \
    "timm" "qwen-vl-utils" "qwen-omni-utils" "keye-vl-utils" \
    "decord" "av" "opencv-python-headless" "librosa" "soundfile" "einops" \
    "ray[default]" "math_verify" "gymnasium" "wandb" "swanlab" "tensorboard" \
    "evalscope>=1.0" "icecream" "py-spy" "pre-commit" \
    "ms-swift[all]@git+https://github.com/modelscope/ms-swift.git@main" && \
    python -m ipykernel install --sys-prefix --name python312 --display-name "Python 3.12 (CUDA 13.3 + vLLM + Swift 4.6)"

# ------------------------------------------------------------------------------
# 7. 针对 A800(8.0) / H20&H200(9.0) / B300(10.0;10.3) 预编译 DeepSpeed 4 大核心算子
#    (fused_adam, cpu_adam, async_io, utils) — 容器启动训练时零 JIT 编译等待！
# ------------------------------------------------------------------------------
RUN ARCH_LIST=$(python -c "import os, torch.utils.cpp_extension as ext; os.environ['TORCH_CUDA_ARCH_LIST']='8.0;9.0;10.0;10.3'; \
      print('8.0;9.0;10.0;10.3') if not hasattr(ext, '_get_cuda_arch_flags') else None" 2>/dev/null || true) && \
    VALID_ARCH=$(python -c "import os, torch.utils.cpp_extension as ext; \
      os.environ['TORCH_CUDA_ARCH_LIST']='8.0;9.0;10.0;10.3'; \
      try: ext._get_cuda_arch_flags(); print('8.0;9.0;10.0;10.3') \
      except Exception: print('8.0;9.0;10.0+PTX')") && \
    echo "Pre-compiling DeepSpeed ops for target architectures: ${VALID_ARCH}" && \
    TORCH_CUDA_ARCH_LIST="${VALID_ARCH}" \
    MAX_JOBS=2 \
    NVCC_THREADS=2 \
    DS_BUILD_OPS=0 \
    DS_BUILD_FUSED_ADAM=1 \
    DS_BUILD_CPU_ADAM=1 \
    DS_BUILD_AIO=1 \
    DS_BUILD_UTILS=1 \
    DS_BUILD_AVX512=0 \
    pip install --no-cache-dir --no-build-isolation --no-deps --force-reinstall --no-binary deepspeed \
      "$(pip freeze | grep '^deepspeed==')" && \
    ds_report

# ------------------------------------------------------------------------------
# 8. 智能安装 SGLang + 预克隆并预编译 Megatron-LM 数据集 C++ 算子 & StartLux-Decision
# ------------------------------------------------------------------------------
RUN (uv pip install --no-cache -c /tmp/torch_constraints.txt "sglang[all]>=0.5.20") || \
    (uv venv /opt/sglang-env --system-site-packages && \
     UV_PROJECT_ENVIRONMENT=/opt/sglang-env uv pip install --no-cache "sglang[all]>=0.5.20") && \
    rm -f /tmp/torch_constraints.txt && \
    mkdir -p /root/.cache/modelscope/_github /opt && \
    git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git /root/.cache/modelscope/_github/Megatron-LM && \
    make -C /root/.cache/modelscope/_github/Megatron-LM/megatron/core/datasets || true && \
    git clone --depth 1 https://github.com/StartLuxLabs/Startlux-Decision.git /opt/StartLux-Decision && \
    printf "set -g mouse on\nset -g history-limit 50000\nset -g default-terminal 'screen-256color'\nset -g status-bg black\nset -g status-fg green\n" > /root/.tmux.conf && \
    printf '#!/bin/bash\nexec jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT:-8888} --no-browser --allow-root --ServerApp.token="${JUPYTER_TOKEN:-}" --ServerApp.password="" --notebook-dir=/workspace\n' > /usr/local/bin/start-jupyter && \
    chmod +x /usr/local/bin/start-jupyter

# 运行期恢复 8 并发编译线程，供大服务器运行使用
ENV MAX_JOBS=8 \
    NVCC_THREADS=4

EXPOSE 8888 8000 30000
WORKDIR /workspace
CMD ["/bin/bash"]
