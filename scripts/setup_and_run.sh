#!/usr/bin/env bash
set -euo pipefail

### ==== 基本路径 ====
PROJECT_ROOT="/ltstorage/home/4xin/MT_Grpo"
# PROJECT_ROOT="/mnt/workspace/xintong/xjf/MT_Grpo" # 取消注释
ENV_YML="$PROJECT_ROOT/scripts/environment.yml"
REQ_TXT="$PROJECT_ROOT/scripts/requirements.txt"
VERL_DIR="$PROJECT_ROOT/verl"
RUN_SCRIPT="$VERL_DIR/custom_grpo_fast_qe.sh"
ENV_NAME="xjf_verl"

export PIP_NO_CACHE_DIR=1  # pip 不使用缓存

echo ">>> 项目根目录: $PROJECT_ROOT"
echo ">>> Conda 环境名: $ENV_NAME"
echo ">>> environment.yml: $ENV_YML"
echo ">>> requirements.txt: $REQ_TXT"
echo ">>> VERL 目录: $VERL_DIR"
echo ">>> 训练脚本: $RUN_SCRIPT"
echo

### ==== 0️⃣ Conda 初始化（关键）====
if ! command -v conda >/dev/null 2>&1; then
  echo "⚠️ conda 命令未加载，尝试自动初始化..."
  if [ -n "${CONDA_HOME:-}" ] && [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_HOME/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "❌ 未找到 conda.sh，请手动设置 CONDA_HOME 或执行 'conda init bash' 后重新登录 shell。"
    exit 1
  fi
else
  if conda shell.bash hook >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ 仍未检测到 conda 命令。"
  exit 1
fi
echo "✅ conda 已可用。"
echo

# ---------- 工具函数：CUDA 检测 & 安装匹配 PyTorch ----------
# 仅用 nvidia-smi 管道获取 CUDA 版本，并按范围映射到 PyTorch 轮子标签
# —— 可靠获取 CUDA 版本（支持老版 nvidia-smi，无 query 字段也行）——
get_cuda_version() {
  local ver=""

  if command -v nvidia-smi >/dev/null 2>&1; then
    # 先扫常见的标题行（前 15 行足够）
    ver="$(LC_ALL=C nvidia-smi 2>/dev/null \
          | head -n 15 \
          | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9.]\+\).*/\1/p' \
          | head -n1)"
    # 再尝试详细模式 -q
    if [ -z "$ver" ]; then
      ver="$(LC_ALL=C nvidia-smi -q 2>/dev/null \
            | sed -n 's/.*CUDA Version[[:space:]]*:[[:space:]]*\([0-9.]\+\).*/\1/p' \
            | head -n1)"
    fi
  fi

  # 最后兜底：系统装了 nvcc 就用它
  if [ -z "$ver" ] && command -v nvcc >/dev/null 2>&1; then
    ver="$(LC_ALL=C nvcc --version \
          | sed -n 's/.*release[[:space:]]\([0-9.]\+\).*/\1/p' \
          | head -n1)"
  fi

  echo "$ver"
}

# —— 将版本映射到 PyTorch 轮子标签 —— 
detect_cuda_tag() {
  local ver="$(get_cuda_version)"
  echo ">>> 检测到 CUDA 版本: ${ver:-<空>}" >&2   # 注意加 >&2

  if [ -z "$ver" ]; then
    echo "cpu"; return
  fi

  # 用 awk 比较浮点
  if awk "BEGIN{exit !($ver >= 12.4)}"; then
    echo "cu124" # 对应 cu12x 系最新版，PyTorch 目前提供 cu124 轮子
  elif awk "BEGIN{exit !($ver >= 12.1)}"; then
    echo "cu121"
  elif awk "BEGIN{exit !($ver >= 11.8)}"; then
    echo "cu118"
  else
    echo "cpu"
  fi
}

install_torch_by_cuda() {
  local tag="$1"
  local TORCH_VER="" VISION_VER="" AUDIO_VER="" INDEX_URL=""

  case "$tag" in
    cu124)
      TORCH_VER="2.6.0"; VISION_VER="0.21.0"; AUDIO_VER="2.6.0"
      INDEX_URL="https://download.pytorch.org/whl/cu124"
      ;;
    cu121)
      TORCH_VER="2.4.0"; VISION_VER="0.19.0"; AUDIO_VER="2.4.0"
      INDEX_URL="https://download.pytorch.org/whl/cu121"
      ;;
    cu118)
      TORCH_VER="2.2.2"; VISION_VER="0.17.2"; AUDIO_VER="2.2.2"
      INDEX_URL="https://download.pytorch.org/whl/cu118"
      ;;
    *)
      echo "⚠️ 未识别到支持的 CUDA 版本，安装 CPU 版 PyTorch..."
      pip install --no-cache-dir "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"
      return
      ;;
  esac

  echo "➡️ 安装 PyTorch (${tag}): torch==${TORCH_VER}  torchvision==${VISION_VER}  torchaudio==${AUDIO_VER}"
  pip install --no-cache-dir --index-url "${INDEX_URL}" \
    "torch==${TORCH_VER}" "torchvision==${VISION_VER}" "torchaudio==${AUDIO_VER}"
}

verify_torch_install() {
  python - <<'PY'
import sys
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("Device name[0]:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch import failed:", e)
    sys.exit(1)
PY
}

install_flashattn_for_torch() {
  # 读取 PyTorch 大小版本
  local pyver FAVER
  pyver="$(python - <<'PY'
import re, torch
v = torch.__version__
m = re.match(r"(\d+)\.(\d+)", v)
print(m.group(0) if m else v)
PY
)"
  echo ">>> Detected torch major.minor = ${pyver}"

  case "$pyver" in
    2.6)  FAVER="2.6.3" ;;
    2.5)  FAVER="2.6.3" ;;   # 多数情况下也可用；不行就改 2.5.9
    2.4)  FAVER="2.5.9" ;;
    2.2)  FAVER="2.3.6" ;;
    *)    FAVER="2.6.3" ;;   # 默认兜底
  esac

  echo ">>> Installing flash-attn==${FAVER}"
  pip uninstall -y flash-attn flash_attn >/dev/null 2>&1 || true
  pip install --no-cache-dir "flash-attn==${FAVER}"
}

# ---------- 1️⃣ 检查环境是否存在 ----------
if conda env list | grep -qE "^\s*${ENV_NAME}\s" ; then
  echo "✅ 检测到环境 ${ENV_NAME} 已存在，跳过创建与安装步骤。"
else
  echo "🚀 环境 ${ENV_NAME} 不存在，开始创建并安装依赖..."
  conda env create -f "${ENV_YML}"

  echo ">>> 激活环境 ${ENV_NAME}"
  conda activate "${ENV_NAME}"

  echo ">>> 1) 自动检测 CUDA 并安装匹配 PyTorch"
  CUDA_TAG="$(detect_cuda_tag)"
  echo ">>> 解析到轮子标签: ${CUDA_TAG}"
  install_torch_by_cuda "$CUDA_TAG"
  echo ">>> 校验 PyTorch 安装"
  verify_torch_install
  echo ">>> 安装 flash-attn 以支持高效训练"
  install_flashattn_for_torch

  # ----------运行库修复 + Torch 动态库导出 + 校验 ----------
  echo ">>> 升级运行库（glibc++），避免 GLIBCXX_3.4.32 缺失"
  # 有些机器旧 libstdc++ 不带 GLIBCXX_3.4.32，先就地升级到新 ABI
  conda install -y -c conda-forge libstdcxx-ng=14.2.0 libgcc-ng=14.2.0 \
    || conda install -y -c conda-forge libstdcxx-ng=13.2.0 libgcc-ng=13.2.0

  # 让动态链接器先能找到 conda 的 lib
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

  # 将 torch 的 lib 目录与 conda 的 lib 一起加入 LD_LIBRARY_PATH（Ray 子进程可继承）
  export TORCH_LIB_DIR="$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
  export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

  # 可选：快速校验（不失败脚本，只打印信息）
  python - <<'PY' || true
import os, importlib, torch, glob, sys
print("[check] torch.__version__      =", torch.__version__)
print("[check] torch.version.cuda     =", torch.version.cuda)
print("[check] torch libdir           =", os.path.join(os.path.dirname(torch.__file__), "lib"))
print("[check] LD_LIBRARY_PATH begins =", (os.environ.get("LD_LIBRARY_PATH") or "")[:180], "...")
try:
    m = importlib.import_module("flash_attn_2_cuda")
    print("[check] flash_attn_2_cuda OK ->", m.__file__)
except Exception as e:
    print("[warn] flash_attn_2_cuda import failed:", e)
PY


  echo ">>> 2) 安装其余依赖（requirements.txt）"
  if [ -f "${REQ_TXT}" ]; then
    # 过滤掉可能误放的 torch/vision/audio，避免与上一步冲突
    TMP_REQ="$(mktemp)"
    grep -Ev '^[[:space:]]*(torch|torchvision|torchaudio)($|[=><])' "${REQ_TXT}" > "${TMP_REQ}" || true
    if [ -s "${TMP_REQ}" ]; then
      pip install --no-cache-dir -r "${TMP_REQ}"
    else
      echo "ℹ️ requirements.txt 中除了 torch 三件套没有其他内容，跳过。"
    fi
    rm -f "${TMP_REQ}"
  else
    echo "⚠️ 未找到 ${REQ_TXT}，跳过其余依赖安装。"
  fi

  echo ">>> 3) 安装 verl (editable mode + vllm,gpu)"
  cd "${VERL_DIR}"
  pip install --no-deps -e .[vllm,gpu]
fi

# ---------- 2️⃣ 激活环境 ----------
echo ">>> 激活环境 ${ENV_NAME}"
conda activate "${ENV_NAME}"
echo "✅ 成功激活环境 ${ENV_NAME}"
# —— 确保运行时能找到 PyTorch 与 conda 的动态库（环境已存在时也需要）——
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_LIB_DIR="$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

### ==== 3️⃣ 执行训练脚本 ====
echo ">>> 开始执行训练脚本：$RUN_SCRIPT"
cd "$VERL_DIR"
bash "$RUN_SCRIPT"
echo "✅ 训练脚本执行完成。"

############################################
# 4️⃣ 打包日志与配置上传用（Overleaf 友好）
############################################
echo ">>> 打包 runs/、wandb/、verl/outputs/ ..."

# 项目根目录：以当前脚本所在目录上一级作为根（scripts/ -> ..）
PROJECT_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
EXPORT_DIR="${PROJECT_ROOT}/exports"
mkdir -p "${EXPORT_DIR}"

cd "${PROJECT_ROOT}"
TO_PACK=()
[ -d "runs" ] && TO_PACK+=("runs")
[ -d "wandb" ] && TO_PACK+=("wandb")
[ -d "verl/outputs" ] && TO_PACK+=("verl/outputs")

if [ ${#TO_PACK[@]} -eq 0 ]; then
  echo "⚠️ 未找到可打包目录（runs / wandb / verl/outputs 均不存在），跳过打包。"
else
  TS="$(date +%Y%m%d_%H%M%S)"
  ARCHIVE="${EXPORT_DIR}/MT_Grpo_logs_${TS}.tar.gz"
  tar -czf "${ARCHIVE}" "${TO_PACK[@]}"
  SIZE_MB="$(du -m "${ARCHIVE}" | awk '{print $1}')"
  echo "✅ 生成压缩包：${ARCHIVE} (${SIZE_MB} MB)"
  if [ "${SIZE_MB}" -gt 45 ]; then
    echo "📦 压缩包 > 45MB，自动分卷以便 Overleaf 上传..."
    split -b 45M -d -a 2 "${ARCHIVE}" "${ARCHIVE}.part-"
    echo "🧩 分卷完成：$(ls -1 ${ARCHIVE}.part-* | wc -l) 个分卷"
    echo "ℹ️ 本地合并：cat ${ARCHIVE}.part-* > ${ARCHIVE}"
  fi
  echo "🎯 打包完成，导出目录：${EXPORT_DIR}"
fi