# llm-docker-builder

构建 LLM 训练与推理一体的 Docker 镜像，目标卡型 A800 / H20 / H200 / B300。
GitHub Actions 构建后推送到 GHCR，并导出单个 `.tar.gz` 上传到 HuggingFace Dataset。

## 镜像内容

| 类别 | 内容 |
| --- | --- |
| 基础镜像 | `nvcr.io/nvidia/cuda:13.3.0-devel-ubuntu24.04`（含 nvcc 13.3） |
| Python | 3.12，由 miniforge 安装到 `/opt/conda` |
| 推理 | vLLM |
| 训练 | ms-swift、DeepSpeed（构建时预编译 fused_adam / cpu_adam / aio / utils 四个算子）、Megatron-LM |
| 量化 | gptqmodel 5.7.0（GPTQ）、torchao（FP8/INT4）、hqq、gguf、bitsandbytes（QLoRA）；推理侧量化内核由 vllm 覆盖。llmcompressor 因与 transformers 5.x 不兼容暂未加入 |
| 决策模型 | netcal + torchmetrics（校准度量 ECE/Brier）、system-one 客户端。基座 Qwen3.5/3.8 已由 ms-swift、transformers（`qwen3_5`）与 vllm 支持，决策头（pointer head）与温度校准需自备训练代码 |
| 注意力 | flash-attn，取自 mjun0812 的预编译轮子仓库（排除 ROCm 构建，优先匹配与 torch 一致的 CUDA 小版本） |
| 前端工具 | JupyterLab（`start-jupyter`）：git 面板、内存表、LSP 代码智能、单元格耗时、ipympl、nvdashboard（GPU 面板）、ipydatagrid、anywidget、itables、plotly、bokeh；jupyter-server-proxy + jupyterlab-iframe + tornado-proxy-handlers + ipylab（在 Jupyter 里嵌入外部 Web UI）；VS Code 浏览器版（`start-code-server`，端口 8080）；Node.js、Bun、npm |
| 命令行 | tmux、vim、htop、nvtop、fzf、jq、yq、bat、ffmpeg、git-lfs、gh、rsync；rg/fzf/zoxide/lazygit 取上游最新静态二进制 |
| Shell | zsh（系统级 `/etc/zsh/zz-image.zsh`，不依赖某个用户的 HOME；Jupyter 终端也强制 zsh）：fzf-tab、zsh-autosuggestions、zsh-syntax-highlighting、`z <目录关键词>`、fzf 快捷键。bash 仍用来跑脚本 |
| 环境管理 | 基座由 conda 提供解释器、uv 装全部依赖；`activate` 列出/切换 `/opt/envs` 下的 venv；`check-envs` 逐个验证基座与各 venv 的导入 |
| Coding Agent | herdr、pi、qmd、ck。pi 预置 9 个扩展（装在 `/opt/pi-extensions`，`/opt/pi-agent/settings.json` 以绝对路径引用，不依赖 HOME 和运行时 npm 源）和 7 个技能（都在 `/opt/pi-agent/skills`）。配置目录由 `PI_CODING_AGENT_DIR=/opt/pi-agent` 固定，不写模型、思考强度和凭据。Python 语言服务器为 ruff 与 ty |
| 中文字体 | Noto Sans CJK + Noto Color Emoji，matplotlib 默认字体已配好，图表中文不再是方框 |
| 独立环境 | `/opt/envs/sglang`（sglang 0.5.21）、`/opt/envs/label-studio`。两者与基座包版本互斥，故各自成 venv：sglang 与 vllm 有多个同名包版本不兼容，label-studio 要 openai<2 与 pyarrow<25，而基座按 vllm 的要求是 openai>=2.25、pyarrow 26，装进基座会弄坏 vllm；两者的 torch 等兼容部分复用基座 |
| 其他 | `/opt/StartLux-Decision`、`/root/.cache/modelscope/_github/Megatron-LM` |

暴露端口 `8888`（Jupyter）、`8000`（vLLM）、`8080`（code-server）、`8090`（label-studio）。默认工作目录 `/workspace`。
容器内 `python` 直接就是 `/opt/conda` 的 base 环境，无需 `conda activate`；SSH 登录会话由 `/etc/profile.d/zz-conda-path.sh` 保证同样的 PATH。

## 版本策略

Python 库版本一律写成 `>=`，上游发布新版后重跑 workflow 即可取得，不需要改文件。
需要固定的工具链版本集中在 Dockerfile 顶部的 `ARG`：

```dockerfile
ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.3.0-devel-ubuntu24.04
ARG MINIFORGE_VERSION=24.11.3-0   # 决定 base 环境的 Python 版本，24.11.3-0 对应 3.12
ARG PYTHON_VERSION=3.12
ARG NODE_VERSION=v22.23.3         # npm 12 要求 node >= 22.22.2
ARG NPM_MAJOR=12
```

本地构建时覆盖：

```bash
docker build --build-arg NODE_VERSION=v24.21.0 -t llm-env:test .
```

以下几处是有意为之，不要随手改动：

- `deepspeed>=0.19.5`。0.19.5 才包含 PR#8238，把 host cxx 编译选项改为 `-std=c++20`。
  vLLM 固定 `torch==2.13.0`，DeepSpeed 0.18.x 在这个 torch 版本上编译算子会失败。
- `datasets`、`peft`、`trl`、`transformers`、`modelscope` 不写版本上限。ms-swift 的包元数据
  已经声明了精确区间，重复写一份只会与上游冲突；把 vllm 一起列入安装命令，让 uv 同时
  满足 vllm 对 transformers 的上限。
- `audioread` 显式安装。`qwen-omni-utils` 的代码直接 `import audioread`，但包元数据没有
  声明这个依赖，过去靠 librosa 传递提供；librosa 1.0.0 起不再依赖 audioread。

Python 版本升级前，先确认 flash-attn 预编译轮子和 DeepSpeed 源码编译支持目标版本。
Dockerfile 第 2 步会校验实际装出的 Python 版本，不一致时立即失败。

## 镜像 tag

构建结束时第 8 步会先跑 `/usr/local/bin/verify-imports.py`（镜像内保留该脚本，可随时重跑），
它会一次列出全部导入失败的模块，而不是遇到第一个就退出。通过后把实际版本写入镜像内的
`/opt/versions.env`：

```
CU=13.3          # 取自 nvcc，不浮动
PY=3.12          # 由 MINIFORGE_VERSION 固定
VLLM=0.32.0      # 以下五项随上游浮动，这里是某次构建的值，以镜像内为准
SWIFT=4.6.0
TORCH=2.13.0
DEEPSPEED=0.19.8
FLASH_ATTN=2.8.3
PI=<pi --version>  # 不进 tag，只在这个文件里溯源
```

workflow 读取该文件生成 tag，形如：

```
cu133-py312-vllm0.32.0-swift4.6.0-rl
```

`workflow_dispatch` 的 `image_tag` 留空即自动派生，填写后使用指定值。

`FLASH_ATTN=none` 表示没匹配到预编译轮子。
GHCR 地址为 `ghcr.io/<owner>/llm-env:<tag>`，HuggingFace 上的文件名为 `llm-env-<tag>.tar.gz`。

## 触发构建

Actions → `Build Image & Upload Single tar.gz to HuggingFace` → Run workflow。

| 输入 | 说明 |
| --- | --- |
| `image_tag` | 留空自动派生 |
| `hf_repo_id` | HuggingFace Dataset 仓库，默认 `TPAMI/llm-docker-images`，不存在时自动创建。命名空间必须是 **HuggingFace** 用户名，与 GitHub 用户名不一定相同 |
| `hf_private` | 是否设为私有仓库。公开仓库可免登录直链下载 |
| `upload_github_release` | 是否额外切分上传一份到 GitHub Releases 作为备份 |

需要在仓库 Secrets 中配置 `HF_TOKEN`，否则跳过 HuggingFace 上传。
任务超时设为 180 分钟；构建前会清理 runner 磁盘以释放约 105GB 空间。

## 使用镜像

```bash
docker run --gpus all -it --rm \
  -p 8888:8888 -p 8000:8000 \
  -v "$PWD:/workspace" \
  -e JUPYTER_TOKEN=<your-token> \
  ghcr.io/yippp/llm-env:<tag>

# 容器内启动 Jupyter
start-jupyter

# 浏览器版 VS Code（端口 8080；首次启动时 stdout 会打印生成的密码文件路径）
start-code-server
```

在 Jupyter 里访问容器内其他端口上的 Web UI（swanlab、TensorBoard、vLLM 等），无需额外配置：

```
http://<host>:8888/proxy/<端口>/      例如 /proxy/6006/ 访问 TensorBoard
```

`jupyterlab_iframe` 提供 IFrame widget，`ipylab` 可从 notebook 控制 JupyterLab 界面。

# 随时重跑导入检查（--no-versions 避免重写 /opt/versions.env）
verify-imports.py --no-versions
```

Shell 小抄：`z <关键词>` 按历史跳目录，Tab 补全支持模糊筛选，Ctrl-R 搜历史命令，Ctrl-T 搜文件。

环境切换与自检：

```bash
activate            # 列出 /opt/envs 下的环境
activate sglang     # 切进去（用法同 conda activate）
deactivate          # 退回基座
check-envs          # 验证基座与各 venv 的导入，改过一个环境后跑一次
```

`check-envs` 是改动某个环境后的安全网：它会把基座和每个 venv 都验一遍，确认没有互相波及。
label-studio 在 venv 里启动（端口避开 code-server 的 8080）：

```bash
/opt/envs/label-studio/bin/label-studio start --port 8090
```

从 HuggingFace 下载后导入：

```bash
sha256sum -c llm-env-<tag>.tar.gz.sha256
docker load -i llm-env-<tag>.tar.gz
```
