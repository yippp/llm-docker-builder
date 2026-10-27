# ==============================================================================
# GPU 训练镜像 Dockerfile，目标卡型 A800 / H20 / H200
# 基础镜像：NVIDIA CUDA 13.3.0 Devel + Ubuntu 24.04（含 nvcc 13.3）
# 构建时预编译：DeepSpeed 算子 (fused_adam/cpu_adam/aio/utils)、FlashAttn、Megatron
# ==============================================================================
ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.3.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

SHELL ["/bin/bash", "-c"]

# ------------------------------------------------------------------------------
# 版本参数：这里只固定工具链版本；Python 库版本全部用 >= 浮动（见第 4、6、8 步），
# 上游发布新版后重跑 workflow 即可取得，镜像 tag 按镜像内实际版本自动派生。
# 临时覆盖示例：docker build --build-arg NODE_VERSION=v24.21.0 .
# ------------------------------------------------------------------------------
# Python 固定为 3.12，ms-swift 官方建议 3.11 或 3.12。base 环境的 Python 版本由
# miniforge 安装器版本决定：24.11.3-0 对应 Python 3.12，26.7.2-0 对应 Python 3.14.7。
# 升级 Python 前，先确认第 5 步的 flash-attn 预编译轮子和第 7 步的 DeepSpeed
# 源码编译支持该版本。
ARG MINIFORGE_VERSION=24.11.3-0
ARG PYTHON_VERSION=3.12
# npm 12 的 engines 要求 node ^22.22.2 || ^24.15.0 || >=26，因此 node 需 >= 22.22.2。
# NPM_MAJOR 只固定主版本号：npm@12 取 12.x 中的最新版，不会升到 13 导致构建失败。
ARG NODE_VERSION=v22.23.3
ARG NPM_MAJOR=12

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Hong_Kong \
    TORCH_CUDA_ARCH_LIST="8.0;9.0" \
    # gptqmodel 的 setup.py 只认不带前缀的 CUDA_ARCH_LIST（探测顺序：该变量 →
    # nvidia-smi），GitHub runner 没有 GPU 和 nvidia-smi，不设则源码编译直接报错。
    # 架构列表只在这两处维护，第 7 步的 DeepSpeed 编译读同一份 ENV。
    # 需要 B300 时不必加回 10.0;10.3，改成 "8.0;9.0+PTX" 由驱动即时编译即可。
    CUDA_ARCH_LIST="8.0;9.0" \
    CONDA_DIR=/opt/conda \
    PATH=/root/.local/bin:/root/.bun/bin:/opt/conda/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    IMPECCABLE_BIN=/opt/impeccable/impeccable \
    # pi 的配置目录默认是 ~/.pi/agent。平台把 HOME 改掉之后，写在 /root 下的
    # settings.json 和技能都不会被读到。固定到 /opt/pi-agent，与 HOME 无关。
    # 没有 PI_CONFIG_DIR，对应变量是 PI_CODING_AGENT_DIR。要改读持久卷上的用户配置，
    # 把这个变量指过去即可；扩展仍在 /opt/pi-extensions，按绝对路径引用。
    # 这里不写模型、思考强度和凭据，那些是使用者自己的配置。
    PI_CODING_AGENT_DIR=/opt/pi-agent \
    SHELL=/bin/zsh \
    # 关闭 pi 自身的后台联网：pi.dev 的版本检查在内网不可达，模型目录刷新同理。
    # 这两个变量按「非空即生效」判断，恢复联网要 unset 而不是设 0。
    PI_OFFLINE=1 \
    PI_SKIP_VERSION_CHECK=1 \
    UV_HTTP_TIMEOUT=300 \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# ------------------------------------------------------------------------------
# 1. 单次 apt-get 安装系统工具、FFmpeg、libaio-dev (DeepSpeed AIO 必需)、RDMA 库、
#    zsh 及两个补全插件与中文字体（图表和终端的中文不再显示为方框）
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    tmux vim nano htop nvtop iotop lsof tree fzf jq bat \
    ripgrep fd-find \
    zsh zsh-autosuggestions zsh-syntax-highlighting \
    fonts-noto-cjk fonts-noto-color-emoji \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev libsndfile1-dev libsox-dev \
    git git-lfs curl wget unzip zip tar xz-utils zstd rsync \
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
    curl -fsSL https://api.github.com/repos/cli/cli/releases/latest -o /tmp/gh.json && \
    GH_URL=$(jq -r '.assets[].browser_download_url | select(endswith("linux_amd64.tar.gz"))' /tmp/gh.json | head -n 1) && \
    test -n "$GH_URL" && curl -fsSL "$GH_URL" -o /tmp/gh.tgz && \
    tar -xzf /tmp/gh.tgz -C /tmp && \
    mv /tmp/gh_*_linux_amd64/bin/gh /usr/local/bin/gh && \
    rm -rf /tmp/gh_* /tmp/gh.tgz && gh --version && \
    curl -fsSL https://code-server.dev/install.sh | sh -s -- --method=standalone --prefix=/usr/local && \
    rm -rf /tmp/fetch && mkdir -p /tmp/fetch && \
    for spec in "BurntSushi/ripgrep|ripgrep-VER-x86_64-unknown-linux-musl.tar.gz|rg" \
                "junegunn/fzf|fzf-VER-linux_amd64.tar.gz|fzf" \
                "ajeetdsouza/zoxide|zoxide-VER-x86_64-unknown-linux-musl.tar.gz|zoxide" \
                "jesseduffield/lazygit|lazygit_VER_Linux_x86_64.tar.gz|lazygit"; do \
      repo="${spec%%|*}"; rest="${spec#*|}"; tpl="${rest%%|*}"; bin="${rest##*|}"; \
      tag=$(curl -fsSL -o /dev/null -w '%{url_effective}' "https://github.com/$repo/releases/latest"); tag="${tag##*/}"; \
      ver="${tag#v}"; \
      asset="https://github.com/$repo/releases/download/$tag/$(printf '%s' "$tpl" | sed "s/VER/$ver/")"; \
      echo "拉取 $repo $tag"; \
      rm -rf /tmp/fetch && mkdir -p /tmp/fetch && \
      curl -fsSL "$asset" | tar -xz -C /tmp/fetch && \
      install -m 755 "$(find /tmp/fetch -type f -name "$bin" | head -n 1)" "/usr/local/bin/$bin"; \
    done && \
    rg --version | head -n 1 && fzf --version && zoxide --version && lazygit --version | head -n 1 && \
    git clone --depth 1 https://github.com/Aloxaf/fzf-tab.git /opt/zsh-plugins/fzf-tab

# ------------------------------------------------------------------------------
# 2. 安装 Conda（Python 版本见 PYTHON_VERSION）+ Node.js + Bun，并升级 pip / uv / npm
#    conda 在这里只提供解释器，ML 依赖全部由 uv 安装（第 4、6 步），没有一条 conda install。
#    装完立即 conda clean -afy：清掉安装器留下的包缓存（归档与解压副本），
#    同一 RUN 内删除不占镜像层。
# ------------------------------------------------------------------------------
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniforge.sh && \
    ${CONDA_DIR}/bin/conda clean -afy && \
    ${CONDA_DIR}/bin/conda init bash && \
    echo "conda activate base" >> /root/.bashrc && \
    ${CONDA_DIR}/bin/python -c "import sys; v=f'{sys.version_info[0]}.{sys.version_info[1]}'; assert v=='${PYTHON_VERSION}', f'miniforge ${MINIFORGE_VERSION} 安装的是 Python {v}，与要求的 ${PYTHON_VERSION} 不一致；请同步修改 MINIFORGE_VERSION 与 PYTHON_VERSION'; print('Python', v, 'OK')" && \
    curl -fsSL https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz | tar -xJ --strip-components=1 -C /usr/local && \
    curl -fsSL https://bun.sh/install | bash && \
    pip install --no-cache-dir -U pip setuptools wheel uv && \
    npm install -g npm@${NPM_MAJOR} && npm cache clean --force && \
    node -v && npm -v && bun -v

# ------------------------------------------------------------------------------
# 3. 安装 AI Coding Agent 套件: herdr + pi + BeaconBay/ck + qmd
# ------------------------------------------------------------------------------
RUN curl -fsSL https://herdr.dev/install.sh | sh && \
    npm install -g --ignore-scripts @earendil-works/pi-coding-agent && \
    npm install -g @tobilu/qmd @beaconbay/ck-search && \
    npm cache clean --force && \
    curl -fsSL https://api.github.com/repos/BeaconBay/ck/releases/latest -o /tmp/ck.json || echo "{}" > /tmp/ck.json && \
    CK_URL=$(jq -r '.assets[]? | select(.name | contains("x86_64-unknown-linux-gnu.tar.gz")) | .browser_download_url' /tmp/ck.json | head -n 1) && \
    if [ -n "$CK_URL" ]; then \
      curl -fsSL "$CK_URL" | tar -xz -C /usr/local/bin && chmod +x /usr/local/bin/ck; \
    fi && \
    rm -f /tmp/ck.json

# ------------------------------------------------------------------------------
# 4. 安装 vLLM（>=0.32.0 浮动），并把 torch 系列的实际版本导出为 /tmp/torch_constraints.txt，
#    供第 6、7 步作为约束使用，避免后续安装把 torch 换成与 vLLM 不配对的版本
# ------------------------------------------------------------------------------
RUN uv pip install --no-cache "vllm>=0.32.0" && \
    pip freeze | grep -E "^(torch|triton|torchvision|torchaudio)==" | sed 's/+.*//' > /tmp/torch_constraints.txt

# ------------------------------------------------------------------------------
# 5. 安装预编译的 flash-attn
# mjun0812 的预编译仓库最新的几个 release 全是 ROCm 构建，只取前 10 个 release
# 会全部错过 CUDA 轮子，因此 per_page=100 并排除 rocm；同一个 flash_attn 版本会同时
# 提供 cu130 和 cu132 两个构建，优先取与 torch.version.cuda 完全一致的那个。
# ------------------------------------------------------------------------------
RUN PY_TAG=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')") && \
    TORCH_MM=$(python -c "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))") && \
    CUDA_MM=$(python -c "import torch; v=torch.version.cuda or '13.0'; print(''.join(v.split('.')[:2]))") && \
    CUDA_MAJ=$(python -c "import torch; print((torch.version.cuda or '13.0').split('.')[0])") && \
    echo "查找预编译轮子：Py=${PY_TAG}, torch${TORCH_MM}, cu${CUDA_MM}（无匹配时回退到 cu${CUDA_MAJ}）" && \
    { curl -fsSL "https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases?per_page=100" -o /tmp/fa.json || echo "[]" > /tmp/fa.json; } && \
    FA_CAND=$(jq -r --arg py "$PY_TAG" --arg tm "torch${TORCH_MM}" \
      '.[].assets[]?.browser_download_url | select(endswith("linux_x86_64.whl") and contains($py) and contains($tm) and (contains("rocm") | not))' /tmp/fa.json) && \
    FA_URL=$( { echo "$FA_CAND" | grep "cu${CUDA_MM}" || echo "$FA_CAND" | grep "cu${CUDA_MAJ}" || true; } \
      | sed 's|.*/flash_attn-\([^+]*\)+.*|\1 &|' | sort -V -r | head -n 1 | cut -d' ' -f2-) && \
    if [ -n "$FA_URL" ]; then \
      echo "安装预编译 flash-attn: $FA_URL" && \
      uv pip install --no-cache --no-deps "$FA_URL"; \
    else \
      echo "⚠️  WARNING: 没有匹配的 flash-attn 预编译轮子 (Py=${PY_TAG}, torch${TORCH_MM}, cu${CUDA_MM}/cu${CUDA_MAJ})，镜像内将不含 flash-attn"; \
    fi && \
    rm -f /tmp/fa.json

# ------------------------------------------------------------------------------
# 6. 安装 JupyterLab + ms-swift（>=4.6 浮动）+ VLM / RL / 决策模型依赖
# datasets、peft、trl、transformers、modelscope 不写版本上限：ms-swift 的元数据已经
# 声明了精确区间（datasets<4.8.5,>=4.0、peft<0.21、trl<1.0、transformers<5.20），
# 这里再写一份会与上游声明冲突。vllm 也列入本次解析，使 uv 同时满足 vllm 对
# transformers 的上限，避免装出可以导入但运行时不兼容的 transformers。
# deepspeed 需 >=0.19.5：该版本才包含 PR#8238（host cxx 编译选项改为 -std=c++20）。
# vllm 0.32.0 固定 torch==2.13.0，0.18.x 在第 7 步源码编译算子时会编译失败。
# swanlab 锁在 0.7.19 并启用 dashboard extra：swanlab watch 直接本地看数据，
# 不用再起一个单独的容器跑服务端。swanboard 显式锁 0.1.9b3（与 swanlab 0.7.19
# dashboard extra 的要求一致），依赖只有 fastapi/peewee/uvicorn 等轻量包，
# 与 vllm 栈无冲突。
# protobuf 落在 6.33.x：0.6.5 起就要求 protobuf<7，vllm 排除的是 6.33.0.*~6.33.4.*，
# 6.33.5/6.33.6 可用；已对镜像内 391 个包全量扫描，无一拒绝 protobuf 6.33.6。
# audioread 必须显式安装：qwen-omni-utils 的 v2_5/audio_process.py 直接 import audioread，
# 但它的包元数据只声明了 av、librosa、packaging、pillow、requests，一直靠 librosa
# 传递提供。librosa 1.0.0（2026-08-11）不再依赖 audioread，因此在本文件显式声明。
# gptqmodel 必须钉 5.7.0：5.8.0 起要求 protobuf>=7.34，与 swanlab 0.7.19 的
# protobuf<7 硬冲突，5.7.0 是最后一个兼容 protobuf 6.x 的版本。它同时把 numpy
# 精确钉在 2.2.6，与 vllm 0.32.0 的约束区间有交集，解析可满足。
# llmcompressor 无法加入：其 0.14.0（最新版）声明支持 transformers 5.15–5.17，
# 但代码 import 的 TORCH_INIT_FUNCTIONS 在 transformers 5.0 已移除，声明的整个
# 区间实际都导入失败（已验证 5.15/5.16/5.17 源码）；vllm 0.32.0 又要求
# transformers>=5.10.4，无回退空间。等上游修好后恢复。
# System One 客户端只装 system-one：jev 要求 Python>=3.14（所有版本），
# 而本镜像固定 3.12，uv 解析阶段就会直接报无解。
# ------------------------------------------------------------------------------
RUN uv pip install --no-cache \
    -c /tmp/torch_constraints.txt \
    "vllm>=0.32.0" \
    nvitop ast-grep-cli pybind11 wheel packaging ninja psutil \
    jupyterlab notebook ipykernel ipywidgets \
    jupyterlab-git jupyter-resource-usage jupyterlab-lsp python-lsp-server jupyterlab-execute-time ipympl \
    jupyter-server-proxy jupyterlab-iframe tornado-proxy-handlers ipylab \
    jupyterlab-nvdashboard ipydatagrid anywidget itables plotly bokeh \
    matplotlib pandas seaborn scipy scikit-learn \
    flash-linear-attention \
    transformers modelscope datasets peft trl sentencepiece \
    "deepspeed>=0.19.5" \
    accelerate optimum bitsandbytes mcore-bridge liger-kernel \
    "gptqmodel==5.7.0" torchao hqq gguf \
    netcal torchmetrics system-one \
    gradio huggingface_hub \
    timm qwen-vl-utils qwen-omni-utils keye-vl-utils \
    decord av opencv-python-headless librosa audioread soundfile einops \
    "ray[default]" math_verify gymnasium wandb tensorboard \
    "swanlab[dashboard]==0.7.19" "swanboard==0.1.9b3" \
    evalscope icecream py-spy pre-commit hf_transfer gpustat pycocotools \
    ruff ty \
    "ms-swift>=4.6" \
    "numpy>=2.1.0" "pyarrow>=18.0.0" && \
    PY_TAG=$(python -c "import sys; print(f'python{sys.version_info[0]}{sys.version_info[1]}')") && \
    python -m ipykernel install --sys-prefix --name "$PY_TAG" --display-name "Python ${PYTHON_VERSION} (CUDA + vLLM + ms-swift)"

# ------------------------------------------------------------------------------
# 7. 为 A800(8.0) / H20、H200(9.0) 预编译 DeepSpeed 的 4 个算子
# ------------------------------------------------------------------------------
RUN VALID_ARCH=$(python -c "import os, torch.utils.cpp_extension as ext; ext._get_cuda_arch_flags(); print(os.environ['TORCH_CUDA_ARCH_LIST'])" 2>/dev/null || echo "${TORCH_CUDA_ARCH_LIST}") && \
    DS_PKG=$(pip freeze | grep '^deepspeed==') && \
    echo "Pre-compiling ${DS_PKG} ops for target architectures: ${VALID_ARCH}" && \
    TORCH_CUDA_ARCH_LIST="${VALID_ARCH}" \
    MAX_JOBS=2 \
    NVCC_THREADS=2 \
    DS_BUILD_OPS=0 \
    DS_BUILD_FUSED_ADAM=1 \
    DS_BUILD_CPU_ADAM=1 \
    DS_BUILD_AIO=1 \
    DS_BUILD_UTILS=1 \
    DS_BUILD_AVX512=0 \
    pip install --no-cache-dir --no-build-isolation --no-deps --force-reinstall --no-binary deepspeed "${DS_PKG}" && \
    ds_report

# ------------------------------------------------------------------------------
# 8. 收尾：numpy 兼容 shim、Megatron-LM、start-jupyter、导入检查与版本清单
#    verify-imports.py 一次列出全部导入失败的模块，而不是遇到第一个就退出，
#    避免每漏一个依赖就重跑一次完整构建；该脚本保留在镜像内，可随时重跑。
#    /opt/versions.env 供 workflow 读取以自动派生镜像 tag，库版本更新后 tag 随之变化。
#    SGLang 与 vLLM 在 flashinfer-python 等 6 个包上版本互斥，不能同环境共存。
# ------------------------------------------------------------------------------
RUN SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])") && \
    printf "try:\n    import numpy as np\n    for _k, _v in [('float_', np.float64), ('int_', np.int64), ('bool8', np.bool_), ('uint', np.uint64), ('complex_', np.complex128), ('object_', np.object_), ('str_', np.str_), ('unicode_', np.str_), ('NaN', np.nan), ('PINF', np.inf), ('NINF', -np.inf), ('infty', np.inf), ('Inf', np.inf)]:\n        if not hasattr(np, _k):\n            setattr(np, _k, _v)\nexcept Exception:\n    pass\n" >> "${SITE_PKG}/sitecustomize.py" && \
    printf '\nfont.sans-serif: Noto Sans CJK SC, DejaVu Sans\naxes.unicode_minus: False\n' >> "${SITE_PKG}/matplotlib/mpl-data/matplotlibrc" && \
    rm -f /tmp/torch_constraints.txt && \
    mkdir -p /root/.cache/modelscope/_github /opt && \
    git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git /root/.cache/modelscope/_github/Megatron-LM && \
    make -C /root/.cache/modelscope/_github/Megatron-LM/megatron/core/datasets || true && \
    git clone --depth 1 https://github.com/StartLuxLabs/Startlux-Decision.git /opt/StartLux-Decision && \
    printf "set -g mouse on\nset -g history-limit 50000\nset -g default-terminal 'screen-256color'\nset -g status-bg black\nset -g status-fg green\n" > /root/.tmux.conf && \
    printf '%s\n' \
      '# 已加载过就返回，避免 /etc/zsh/zshrc 和用户 .zshrc 各引一次' \
      '[[ -n ${_LLM_ENV_ZSH:-} ]] && return' \
      '_LLM_ENV_ZSH=1' \
      'HISTFILE="${HOME}/.zsh_history"' \
      'HISTSIZE=50000' \
      'SAVEHIST=50000' \
      'setopt SHARE_HISTORY HIST_IGNORE_DUPS HIST_REDUCE_BLANKS' \
      'autoload -Uz compinit && compinit' \
      'source /opt/zsh-plugins/fzf-tab/fzf-tab.plugin.zsh' \
      'eval "$(fzf --zsh 2>/dev/null)"' \
      'eval "$(zoxide init zsh)"' \
      'source /usr/share/zsh-autosuggestions/zsh-autosuggestions.zsh' \
      'source /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh' \
      > /etc/zsh/zz-image.zsh && \
    touch /etc/zsh/zshrc && \
    grep -q zz-image.zsh /etc/zsh/zshrc || printf '\n[ -r /etc/zsh/zz-image.zsh ] && source /etc/zsh/zz-image.zsh\n' >> /etc/zsh/zshrc && \
    printf '%s\n' '[ -r /etc/zsh/zz-image.zsh ] && source /etc/zsh/zz-image.zsh' > /etc/skel/.zshrc && \
    cp /etc/skel/.zshrc /root/.zshrc && \
    usermod -s /bin/zsh root && \
    printf 'export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}\n' > /etc/profile.d/zz-conda-path.sh && \
    printf '#!/bin/bash\nexec code-server --bind-addr 0.0.0.0:${CODE_SERVER_PORT:-8080} --auth password\n' > /usr/local/bin/start-code-server && \
    chmod +x /usr/local/bin/start-code-server && \
    printf '#!/bin/bash\nexec jupyter lab --ip=0.0.0.0 --port=${JUPYTER_PORT:-8888} --no-browser --allow-root --ServerApp.token="${JUPYTER_TOKEN:-}" --ServerApp.password="" --notebook-dir=/workspace\n' > /usr/local/bin/start-jupyter && \
    chmod +x /usr/local/bin/start-jupyter && \
    mkdir -p /etc/jupyter && \
    printf '%s\n' 'c.ServerApp.terminado_settings = {"shell_command": ["/bin/zsh", "-l"]}' > /etc/jupyter/jupyter_server_config.py && \
    printf '%s\n' \
      '#!/usr/bin/env python3' \
      'import importlib, subprocess, sys' \
      'MODS = """numpy pyarrow scipy sklearn cv2 pandas pydantic openai' \
      'torch torchvision torchaudio transformers datasets peft trl deepspeed' \
      'modelscope vllm swift timm decord librosa audioread soundfile einops' \
      'qwen_vl_utils qwen_omni_utils ray math_verify gymnasium wandb swanlab swanboard hf_transfer gpustat pycocotools' \
      ' gptqmodel torchao hqq gguf jupyter_server_proxy jupyterlab_iframe tornado_proxy_handlers ipylab' \
      ' jupyterlab_nvdashboard ipydatagrid anywidget itables plotly bokeh' \
      ' netcal torchmetrics system_one sentencepiece tensorboard""".split()' \
      'failed = []' \
      'for name in MODS:' \
      '    try:' \
      '        importlib.import_module(name)' \
      '    except Exception as exc:' \
      '        failed.append(f"{name}: {type(exc).__name__}: {exc}")' \
      'if failed:' \
      '    print("以下模块导入失败，共 %d 个:" % len(failed))' \
      '    for item in failed:' \
      '        print("  -", item)' \
      '    sys.exit(1)' \
      'from openai.types.responses import NamespaceTool' \
      'from vllm import LLM, SamplingParams' \
      'import vllm, swift, torch, transformers, deepspeed, pydantic' \
      'print("全部核心模块导入成功 | vllm", vllm.__version__, "| ms-swift", swift.__version__)' \
      'print("torch", torch.__version__, "| transformers", transformers.__version__,' \
      '      "| deepspeed", deepspeed.__version__, "| pydantic", pydantic.__version__)' \
      'try:' \
      '    pi_ver = subprocess.check_output(["pi", "--version"], text=True).strip().split()[-1]' \
      'except Exception as exc:' \
      '    pi_ver = "none"' \
      '    print("pi --version 失败:", exc)' \
      'else:' \
      '    print("pi", pi_ver)' \
      'import importlib.metadata as md' \
      'def _ver(pkg):' \
      '    try:' \
      '        return md.version(pkg)' \
      '    except Exception:' \
      '        return "none"' \
      'if "--no-versions" in sys.argv:' \
      '    sys.exit(0)' \
      'print("已写出 /opt/versions.env:")' \
      'with open("/opt/versions.env", "w") as fh:' \
      '    for key, val in [' \
      '        ("PY", f"{sys.version_info[0]}.{sys.version_info[1]}"),' \
      '        ("VLLM", vllm.__version__),' \
      '        ("SWIFT", swift.__version__),' \
      '        ("TORCH", torch.__version__),' \
      '        ("DEEPSPEED", deepspeed.__version__),' \
      '        ("FLASH_ATTN", _ver("flash-attn")),' \
      '        ("PI", pi_ver),' \
      '    ]:' \
      '        line = key + "=" + str(val)' \
      '        fh.write(line + "\n")' \
      '        print(" ", line)' \
      > /usr/local/bin/verify-imports.py && \
    chmod +x /usr/local/bin/verify-imports.py && \
    python /usr/local/bin/verify-imports.py && \
    echo "CU=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -n 1)" | cat - /opt/versions.env > /tmp/v.env && \
    mv /tmp/v.env /opt/versions.env && cat /opt/versions.env

# ------------------------------------------------------------------------------
# 9. 预置 pi 扩展与技能。扩展在构建期装好、settings.json 用绝对路径引用，
#    好处是确定性：容器内的 npm 默认指向 registry.npmjs.org，不依赖运行环境是否
#    配好了内部镜像源，启动时也不会去检定更新。内网若有 npm 镜像源，可自行用
#    `pi install npm:<包>` 再装；要恢复 pi 自身的后台联网需**取消这两个变量**
#    （pi 判断的是变量是否非空，PI_OFFLINE=0 仍然是开着的），例如：
#    `env -u PI_OFFLINE -u PI_SKIP_VERSION_CHECK pi`。
#    扩展必须选 Node 构建：镜像里的 pi 由 npm 全局安装、跑在 Node 上，Bun 专用
#    构建（opensec-pi-fff / opensec-pi-todo 就是）加载时会抛错并让 pi 直接退出。
#    对应包分别换成 @ff-labs/pi-fff 与 @juicesharp/rpiv-todo，下面另做构建期拦截。
#    凭据文件（auth.json / models.json 等）不写入，由使用者在容器内自行配置。
#    ruff 与 ty 对应 pi-lsp 内置目录里仅有的两个 Python 语言服务器（ruff server /
#    ty server）；python-lsp-server 是第 6 步为 JupyterLab 装的，两者互不干扰。
#    impeccable 技能的引擎平时从 GitHub Releases 下载，这里预先取回并用
#    IMPECCABLE_BIN 指向它，附带 sha256 校验。
#    ui-ux-pro-max 与 ui-styling 由 ui-ux-pro-max-cli 在构建期装入，再挪到
#    /opt/pi-agent/skills。--offline 用包内自带模板与数据，不联网。
#    扩展装在 /opt/pi-extensions，不放进 $HOME：平台重映射 HOME 时，写在
#    /root/.pi 下的绝对路径会静默失效。
# ------------------------------------------------------------------------------
COPY pi/skills/ /opt/pi-agent/skills/
RUN mkdir -p /opt/pi-extensions && \
    printf '%s\n' \
      '{' \
      '  "name": "pi-extensions",' \
      '  "private": true,' \
      '  "dependencies": {' \
      '    "@dietrichgebert/ponytail": "^5.3.2",' \
      '    "@ff-labs/pi-fff": "^0.11.0",' \
      '    "@juicesharp/rpiv-ask-user-question": "^2.13.9",' \
      '    "@juicesharp/rpiv-todo": "^2.13.9",' \
      '    "@narumitw/pi-lsp": "^0.49.8",' \
      '    "pi-background-tasks": "^2.6.10",' \
      '    "pi-goal-x": "^0.32.3",' \
      '    "pi-hermes-memory": "^0.9.11",' \
      '    "pi-powerline-footer": "^0.20.2"' \
      '  }' \
      '}' > /opt/pi-extensions/package.json && \
    cd /opt/pi-extensions && \
    npm install --omit=dev --legacy-peer-deps --no-audit --no-fund && \
    npm cache clean --force && \
    for p in @dietrichgebert/ponytail @ff-labs/pi-fff @juicesharp/rpiv-ask-user-question @juicesharp/rpiv-todo @narumitw/pi-lsp pi-background-tasks pi-goal-x pi-hermes-memory pi-powerline-footer; do \
      test -d "/opt/pi-extensions/node_modules/$p" || { echo "pi 扩展未装成功: $p"; exit 1; }; \
      if grep -rqs "is a Bun build" "/opt/pi-extensions/node_modules/$p"; then \
        echo "$p 是 Bun 专用构建，在 Node 下会加载失败"; exit 1; \
      fi; \
    done && \
    printf '%s\n' \
      '{' \
      '  "theme": "dark",' \
      '  "packages": [' \
      '    "/opt/pi-extensions/node_modules/@dietrichgebert/ponytail",' \
      '    "/opt/pi-extensions/node_modules/@ff-labs/pi-fff",' \
      '    "/opt/pi-extensions/node_modules/@juicesharp/rpiv-ask-user-question",' \
      '    "/opt/pi-extensions/node_modules/@juicesharp/rpiv-todo",' \
      '    "/opt/pi-extensions/node_modules/@narumitw/pi-lsp",' \
      '    "/opt/pi-extensions/node_modules/pi-background-tasks",' \
      '    "/opt/pi-extensions/node_modules/pi-goal-x",' \
      '    "/opt/pi-extensions/node_modules/pi-hermes-memory",' \
      '    "/opt/pi-extensions/node_modules/pi-powerline-footer"' \
      '  ]' \
      '}' > /opt/pi-agent/settings.json && \
    IMP_VER=$(cat /opt/pi-agent/skills/impeccable/scripts/VERSION) && \
    mkdir -p /opt/impeccable && \
    curl -fsSL "https://github.com/pbakaus/impeccable/releases/download/engine-v${IMP_VER}/impeccable-linux-x64" -o /opt/impeccable/impeccable-linux-x64 && \
    curl -fsSL "https://github.com/pbakaus/impeccable/releases/download/engine-v${IMP_VER}/impeccable-linux-x64.sha256" -o /tmp/impeccable.sha256 && \
    ( cd /opt/impeccable && sha256sum -c /tmp/impeccable.sha256 ) && \
    rm -f /tmp/impeccable.sha256 && \
    chmod +x /opt/impeccable/impeccable-linux-x64 && \
    ln -sf /opt/impeccable/impeccable-linux-x64 /opt/impeccable/impeccable && \
    test -x /opt/impeccable/impeccable && echo "impeccable 引擎已就位 (${IMP_VER})" && \
    npm install -g ui-ux-pro-max-cli && \
    uipro init --ai universal --global --offline && \
    for s in ui-ux-pro-max ui-styling; do \
      test -f "/root/.agents/skills/$s/SKILL.md" || { echo "uipro 未装入 $s"; exit 1; }; \
      rm -rf "/opt/pi-agent/skills/$s"; \
      mv "/root/.agents/skills/$s" "/opt/pi-agent/skills/$s"; \
    done && \
    rm -rf /root/.agents && \
    npm cache clean --force && \
    echo "pi 技能: 仓库内 5 个 + ui-ux-pro-max / ui-styling（都在 /opt/pi-agent/skills）"

# ------------------------------------------------------------------------------
# 10. 独立环境（一）：sglang。它无法与基座共存：两边各自钉死多个同名包的不同
#     版本（transformers / xgrammar / flashinfer-python / tokenizers /
#     huggingface-hub / compressed-tensors / cuda-tile 等），而一个环境里同名包
#     只能存在一份，实际覆盖掉几个取决于 pip 对已满足约束的包会保留。
#     两者最贵的 torch 2.13.0 与 torchaudio 2.11.0 钉版本恰好一致，于是用
#     venv --system-site-packages 复用基座这两项，只把不兼容的包装进 venv。
#     必须用 pip 而不是 uv：pip 尊重 --system-site-packages（只装缺的），
#     并且拒绝卸载 venv 之外的包——基座因此不会被改坏；uv 这两点都不支持。
# ------------------------------------------------------------------------------
RUN /opt/conda/bin/python -m venv --system-site-packages /opt/envs/sglang && \
    /opt/envs/sglang/bin/pip install --no-cache-dir "sglang==0.5.21" && \
    printf 'sglang\n' > /opt/envs/sglang/.check-imports && \
    /opt/envs/sglang/bin/python -c "import sglang, torch, transformers, importlib.metadata as md; print('sglang', sglang.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__, '| flashinfer', md.version('flashinfer-python'))"

# ------------------------------------------------------------------------------
# 11. 独立环境（二）：label-studio。它要求 openai<2.0 与 pyarrow<25，而基座（按
#     vllm 的要求）是 openai>=2.25、pyarrow 26，装进基座会把 vllm 弄坏，所以单独
#     建 venv。另有 python-json-logger / pytz / setuptools 等更严的钉，一并隔离。
#     同样用 --system-site-packages 复用兼容的部分，冲突的包只落在 venv 里。
#     它默认监听 8080，与 code-server 撞车，启动时用 --port 8090。
# ------------------------------------------------------------------------------
RUN /opt/conda/bin/python -m venv --system-site-packages /opt/envs/label-studio && \
    /opt/envs/label-studio/bin/pip install --no-cache-dir -U label-studio && \
    printf 'label_studio\n' > /opt/envs/label-studio/.check-imports && \
    /opt/envs/label-studio/bin/python -c "import label_studio, importlib.metadata as md; print('label-studio', md.version('label-studio'), '| openai', md.version('openai'), '| pyarrow', md.version('pyarrow'))"

# ------------------------------------------------------------------------------
# 12. 环境切换与自检。activate <名字> 列出或切换 /opt/envs 下的环境（用法同
#     conda activate）；check-envs 依次验证基座与各 venv，改动其中一个环境后
#     跑一次就能确认另一个没被波及。注意 check-envs 调用 verify-imports.py 时
#     必须带 --no-versions：那个脚本默认会重写 /opt/versions.env，而在第 8 步
#     之后该文件已经被拼上了 CU= 一行，重写会把它抹掉，使 workflow 的 tag 派生
#     步骤报“缺字段 CU”而失败。
# ------------------------------------------------------------------------------
RUN printf '%s\n' \
      '# 切换 /opt/envs 下的 venv：activate 列表，activate <名字> 进入' \
      'activate() {' \
      '  local base=/opt/envs' \
      '  if [ -z "$1" ]; then ls -1 "$base" 2>/dev/null || echo "还没有创建任何环境"; return 0; fi' \
      '  if [ ! -f "$base/$1/bin/activate" ]; then echo "没有名为 $1 的环境"; ls -1 "$base" 2>/dev/null; return 1; fi' \
      '  source "$base/$1/bin/activate"' \
      '}' > /etc/profile.d/zz-activate.sh && \
    printf '%s\n' \
      '#!/bin/bash' \
      '# 逐个验证基座与 /opt/envs 下各环境的导入是否正常' \
      'fail=0' \
      'echo "== 基座 /opt/conda =="' \
      'if python /usr/local/bin/verify-imports.py --no-versions > /tmp/check-envs-base.log 2>&1; then' \
      '  echo "  导入检查通过"' \
      'else' \
      '  echo "  导入检查失败："' \
      '  tail -n 5 /tmp/check-envs-base.log | sed "s/^/    /"' \
      '  fail=1' \
      'fi' \
      'for d in /opt/envs/*/; do' \
      '  [ -d "$d" ] || continue' \
      '  name=$(basename "$d")' \
      '  echo "== $name =="' \
      '  if [ -f "$d/.check-imports" ]; then' \
      '    while read -r mod; do' \
      '      [ -z "$mod" ] && continue' \
      '      if "$d/bin/python" -c "import $mod" 2>/dev/null; then' \
      '        echo "  import $mod: 正常"' \
      '      else' \
      '        echo "  import $mod: 失败"' \
      '        fail=1' \
      '      fi' \
      '    done < "$d/.check-imports"' \
      '  else' \
      '    echo "  未声明检查项（$d.check-imports）"' \
      '  fi' \
      'done' \
      'exit $fail' > /usr/local/bin/check-envs && \
    chmod +x /usr/local/bin/check-envs && \
    printf '\nsource /etc/profile.d/zz-activate.sh\n' >> /etc/zsh/zz-image.zsh && \
    touch /etc/bash.bashrc && \
    grep -q zz-activate.sh /etc/bash.bashrc || printf '\n[ -r /etc/profile.d/zz-activate.sh ] && . /etc/profile.d/zz-activate.sh\n' >> /etc/bash.bashrc && \
    check-envs

ENV MAX_JOBS=8 \
    NVCC_THREADS=4

EXPOSE 8888 8000 8080 8090
WORKDIR /workspace
CMD ["/bin/bash"]
