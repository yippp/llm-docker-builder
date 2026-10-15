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
| 量化 | gptqmodel 5.7.0（GPTQ）、llmcompressor（FP8/INT8 → compressed-tensors）、torchao、hqq、gguf、bitsandbytes（QLoRA）；推理侧量化内核由 vllm 覆盖 |
| 注意力 | flash-attn，取自 mjun0812 的预编译轮子仓库（排除 ROCm 构建，优先匹配与 torch 一致的 CUDA 小版本） |
| 前端工具 | JupyterLab（`start-jupyter`，含 git 面板、内存表、LSP 代码智能、单元格耗时、ipympl 五个插件）、VS Code 浏览器版（`start-code-server`，端口 8080）、Node.js、Bun、npm |
| 命令行 | tmux、vim、htop、nvtop、fzf、jq、yq、bat、ffmpeg、git-lfs、gh、rsync；rg/fzf/zoxide/lazygit 取上游最新静态二进制 |
| Shell | zsh（root 默认，bash 保留跑脚本）：fzf-tab Tab 模糊补全、zsh-autosuggestions 灰色历史提示、zsh-syntax-highlighting 语法高亮、`z <目录关键词>` 按历史跳转、fzf 快捷键 |
| 中文字体 | Noto Sans CJK + Noto Color Emoji，matplotlib 默认字体已配好，图表中文不再是方框 |
| Coding Agent | herdr、pi、qmd、ck |
| 其他 | `/opt/StartLux-Decision`、`/root/.cache/modelscope/_github/Megatron-LM` |

暴露端口 `8888`（Jupyter）、`8000`（vLLM）、`8080`（code-server）。默认工作目录 `/workspace`。
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
  vLLM 0.31.0 固定 `torch==2.13.0`，DeepSpeed 0.18.x 在这个 torch 版本上编译算子会失败。
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
CU=13.3
PY=3.12
VLLM=0.31.0
SWIFT=4.6.0
TORCH=2.13.0
DEEPSPEED=0.19.7
FLASH_ATTN=2.8.3
```

workflow 读取该文件生成 tag，形如：

```
cu133-py312-vllm0.31.0-swift4.6.0-rl
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

# 随时重跑导入检查
verify-imports.py
```

Shell 小抄：`z <关键词>` 按历史跳目录，Tab 补全支持模糊筛选，Ctrl-R 搜历史命令，Ctrl-T 搜文件。

从 HuggingFace 下载后导入：

```bash
sha256sum -c llm-env-<tag>.tar.gz.sha256
docker load -i llm-env-<tag>.tar.gz
```
