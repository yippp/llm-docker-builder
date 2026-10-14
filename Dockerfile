# ==============================================================================
# Base: NVIDIA CUDA 13.3.0 Devel + Ubuntu 24.04 (自带 nvcc 13.3)
# 集成: Conda(Py3.12) + JupyterLab/Notebook + vLLM 0.31 + SGLang + ms-swift 4.6
#       + FlashAttn(预编译) + herdr + pi-agent + BeaconBay/ck + qmd + rg/fd/tmux/nvitop
# ==============================================================================
FROM nvcr.io/nvidia/cuda:13.3.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Hong_Kong \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0" \
    CONDA_DIR=/opt/conda \
    PATH=/root/.local/bin:/root/.bun/bin:/opt/conda/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=300 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# ------------------------------------------------------------------------------
# 1. 系统层工具: tmux, rg, fd, fzf, bat, jq, RDMA多卡通信库 + CUDA 13.3 向下兼容包
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux vim nano htop nvtop iotop lsof tree fzf jq bat \
    ripgrep fd-find \
    git git-lfs curl wget unzip zip tar xz-utils zstd \
    net-tools iproute2 iputils-ping openssh-client openssh-server sudo ca-certificates gnupg \
    build-essential cmake ninja-build pkg-config \
    libnuma-dev libaio-dev libssl-dev libffi-dev libgl1 libglib2.0-0 \
    libibverbs-dev rdma-core ibverbs-utils \
    cuda-compat-13-3 \
    && git lfs install \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && ln -sf /usr/bin/batcat /usr/local/bin/bat \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 安装 GitHub CLI (gh) 与 yq
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update && apt-get install -y gh \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 \
    && chmod +x /usr/local/bin/yq

# ------------------------------------------------------------------------------
# 2. 安装 Miniforge3 (Conda + Mamba, Python 3.12) 与 Node.js >= 22.19 (pi agent 要求)
# ------------------------------------------------------------------------------
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniforge.sh \
    && ${CONDA_DIR}/bin/mamba install -y -c conda-forge \
        python=3.12 \
        "nodejs>=22.19" \
        pip \
    && ${CONDA_DIR}/bin/mamba clean -ya \
    && ${CONDA_DIR}/bin/conda init bash \
    && echo "conda activate base" >> /root/.bashrc

# 安装 Bun 运行时
RUN curl -fsSL https://bun.sh/install | bash

# ------------------------------------------------------------------------------
# 3. 安装 AI Coding Agent 与检索工具: herdr + pi + BeaconBay/ck + qmd + ast-grep
# ------------------------------------------------------------------------------
RUN curl -fsSL https://herdr.dev/install.sh | sh

RUN npm install -g --ignore-scripts @earendil-works/pi-coding-agent \
    && npm install -g @tobilu/qmd @beaconbay/ck-search \
    && npm cache clean --force

# 拉取 BeaconBay/ck 原生 Linux Rust 二进制包加速语义检索
RUN python -c '
import json, urllib.request, tarfile, io, os
try:
    req = urllib.request.Request("https://api.github.com/repos/BeaconBay/ck/releases/latest", headers={"User-Agent": "Docker"})
    data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    for asset in data.get("assets", []):
        if "x86_64-unknown-linux-gnu.tar.gz" in asset["name"]:
            tar_bytes = urllib.request.urlopen(asset["browser_download_url"], timeout=60).read()
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
                tar.extractall("/usr/local/bin")
            os.chmod("/usr/local/bin/ck", 0o755)
            break
except Exception as e:
    print("Keep npm ck binary as fallback:", e)
'

# ------------------------------------------------------------------------------
# 4. 安装 uv, nvitop, ast-grep 以及 JupyterLab / Jupyter Notebook 全家桶
# ------------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    uv nvitop ast-grep-cli \
    jupyterlab notebook ipykernel ipywidgets \
    matplotlib pandas seaborn scipy \
    && python -m ipykernel install --sys-prefix --name python312 --display-name "Python 3.12 (CUDA 13.3 + vLLM + Swift)"

# ------------------------------------------------------------------------------
# 5. 安装 vLLM 0.31.0 与 PyTorch 底座，并生成版本锁保护文件
# ------------------------------------------------------------------------------
RUN uv pip install --no-cache \
    "vllm==0.31.0" \
    --torch-backend=cu130 \
    && pip freeze | grep -E "^(torch|triton|torchvision|torchaudio)==" > /tmp/torch_constraints.txt

# ------------------------------------------------------------------------------
# 6. 自动匹配并安装预编译 Flash-Attention 2 Wheel (零源码编译，耗时 <15秒)
# ------------------------------------------------------------------------------
RUN python -c '
import torch, sys, json, urllib.request, subprocess

py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi_str = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
cuda_major = torch.version.cuda.split(".")[0] if torch.version.cuda else "13"

print(f"Looking for prebuilt flash-attn wheel: Py={py_tag}, Torch={torch_mm}, CUDA_major={cuda_major}, ABI={abi_str}")

repos = [
    "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases?per_page=10",
    "https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=5"
]

wheel_url = None
for api_url in repos:
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Docker-Build"})
        releases = json.loads(urllib.request.urlopen(req, timeout=20).read().decode())
        for rel in releases:
            for asset in rel.get("assets", []):
                name = asset["name"]
                if (name.endswith("linux_x86_64.whl") and py_tag in name
                    and f"torch{torch_mm}" in name and f"cu{cuda_major}" in name):
                    wheel_url = asset["browser_download_url"]
                    break
            if wheel_url:
                break
    except Exception as e:
        print(f"Warning querying {api_url}: {e}")
    if wheel_url:
        break

if wheel_url:
    print(f"Found exact prebuilt wheel: {wheel_url}")
    subprocess.run(["uv", "pip", "install", "--no-cache", wheel_url], check=True)
else:
    print("Exact cu13 wheel not found via API, falling back to binary-only find-links...")
    subprocess.run([
        "uv", "pip", "install", "--no-cache", "--only-binary=:all:", "flash-attn",
        "--find-links", "https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.3"
    ], check=False)
'

# ------------------------------------------------------------------------------
# 7. 安装 SGLang + ms-swift 4.6 + Transformers + ModelScope (严格遵守 Torch 版本锁)
# ------------------------------------------------------------------------------
RUN uv pip install --no-cache \
    -c /tmp/torch_constraints.txt \
    "sglang[all]>=0.5.20"

RUN uv pip install --no-cache \
    -c /tmp/torch_constraints.txt \
    "transformers>=5.10.4" \
    "modelscope>=1.40.0" \
    "huggingface_hub[cli,hf_transfer]" \
    "accelerate" "peft" "deepspeed" "bitsandbytes" \
    "liger-kernel" "evalscope" "datasets" "qwen-vl-utils" \
    "ms-swift@git+https://github.com/modelscope/ms-swift.git@main" \
    && rm -f /tmp/torch_constraints.txt

# ------------------------------------------------------------------------------
# 8. 预置 Megatron-LM、tmux 配置与一键启动 Jupyter 脚本 (start-jupyter)
# ------------------------------------------------------------------------------
RUN mkdir -p /root/.cache/modelscope/_github && \
    git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git /root/.cache/modelscope/_github/Megatron-LM

RUN printf "set -g mouse on\nset -g history-limit 50000\nset -g default-terminal 'screen-256color'\nset -g status-bg black\nset -g status-fg green\n" > /root/.tmux.conf

RUN printf '#!/bin/bash\nexec jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT:-8888} --no-browser --allow-root --ServerApp.token="${JUPYTER_TOKEN:-}" --ServerApp.password="" --notebook-dir=/workspace\n' > /usr/local/bin/start-jupyter \
    && chmod +x /usr/local/bin/start-jupyter

EXPOSE 8888 8000 30000
WORKDIR /workspace
CMD ["/bin/bash"]
