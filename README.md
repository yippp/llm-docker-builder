# llm-docker-builder

构建 LLM 训练与推理一体的 Docker 镜像，目标卡型 A800 / H20 / H200 / B300。
GitHub Actions 构建后推送到 GHCR，并导出单个 `.tar.gz` 上传到 HuggingFace Dataset。

## 镜像内容

| 类别 | 内容 |
| --- | --- |
| 基础镜像 | `nvcr.io/nvidia/cuda:13.3.0-devel-ubuntu24.04`（含 nvcc 13.3） |
| Python | 3.12，由 miniforge 安装到 `/opt/conda` |
| 推理 | vLLM（base 环境）、SGLang（`/opt/sglang-env`，入口 `sglang-python`） |
| 训练 | ms-swift、DeepSpeed（构建时预编译 fused_adam / cpu_adam / aio / utils 四个算子）、Megatron-LM |
| 注意力 | flash-attn、causal-conv1d，取自上游预编译轮子；匹配不到时跳过并打印 WARNING |
| 前端工具 | JupyterLab（`start-jupyter`）、Node.js、Bun、npm |
| 命令行 | tmux、vim、htop、nvtop、fzf、jq、yq、bat、ripgrep、fd、ffmpeg、git-lfs、gh |
| Coding Agent | herdr、pi、qmd、ck |
| 其他 | `/opt/StartLux-Decision`、`/root/.cache/modelscope/_github/Megatron-LM` |

暴露端口 `8888`（Jupyter）、`8000`（vLLM）、`30000`（SGLang）。默认工作目录 `/workspace`。

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

两处版本约束是有意保留的，不要随手放宽：

- `deepspeed>=0.19.5`。0.19.5 才包含 PR#8238，把 host cxx 编译选项改为 `-std=c++20`。
  vLLM 0.31.0 固定 `torch==2.13.0`，DeepSpeed 0.18.x 在这个 torch 版本上编译算子会失败。
- `datasets`、`peft`、`trl`、`transformers`、`modelscope` 不写版本上限。ms-swift 的包元数据
  已经声明了精确区间，重复写一份只会与上游冲突；把 vllm 一起列入安装命令，让 uv 同时
  满足 vllm 对 transformers 的上限。

Python 版本升级前，先确认 flash-attn 预编译轮子和 DeepSpeed 源码编译支持目标版本。
Dockerfile 第 2 步会校验实际装出的 Python 版本，不一致时立即失败。

## 镜像 tag

构建结束时第 8 步会把实际版本写入镜像内的 `/opt/versions.env`：

```
CU=13.3
PY=3.12
VLLM=0.31.0
SWIFT=4.6.0
TORCH=2.13.0
DEEPSPEED=0.19.7
SGLANG=0.5.21
FLASH_ATTN=2.8.3.post2
```

workflow 读取该文件生成 tag，形如：

```
cu133-py312-vllm0.31.0-swift4.6.0-sglang0.5.21-rl
```

`workflow_dispatch` 的 `image_tag` 留空即自动派生，填写后使用指定值。
GHCR 地址为 `ghcr.io/<owner>/llm-env:<tag>`，HuggingFace 上的文件名为 `llm-env-<tag>.tar.gz`。

## 触发构建

Actions → `Build Image & Upload Single tar.gz to HuggingFace` → Run workflow。

| 输入 | 说明 |
| --- | --- |
| `image_tag` | 留空自动派生 |
| `hf_repo_id` | HuggingFace Dataset 仓库，默认 `yippp/llm-docker-images`，不存在时自动创建 |
| `hf_private` | 是否设为私有仓库。公开仓库可免登录直链下载 |
| `upload_github_release` | 是否额外切分上传一份到 GitHub Releases 作为备份 |

需要在仓库 Secrets 中配置 `HF_TOKEN`，否则跳过 HuggingFace 上传。
任务超时设为 180 分钟；构建前会清理 runner 磁盘以释放约 105GB 空间。

## 使用镜像

```bash
docker run --gpus all -it --rm \
  -p 8888:8888 -p 8000:8000 -p 30000:30000 \
  -v "$PWD:/workspace" \
  -e JUPYTER_TOKEN=<your-token> \
  ghcr.io/yippp/llm-env:<tag>

# 容器内启动 Jupyter
start-jupyter

# SGLang 使用独立环境
sglang-python -m sglang.launch_server --help
```

从 HuggingFace 下载后导入：

```bash
sha256sum -c llm-env-<tag>.tar.gz.sha256
docker load -i llm-env-<tag>.tar.gz
```
