#!/usr/bin/env bash
set -euo pipefail

### ==== 基本路径 ====
PROJECT_ROOT="/ltstorage/home/4xin/MT_Grpo"
# PROJECT_ROOT="/mnt/workspace/xintong/xjf/MT_Grpo" # 取消注释
ENV_YML="$PROJECT_ROOT/scripts/environment.yml"
VERL_DIR="$PROJECT_ROOT/verl"
RUN_SCRIPT="$VERL_DIR/custom_grpo_fast_qe.sh"
ENV_NAME="xjf_verl"

export PIP_NO_CACHE_DIR=1  # pip 不使用缓存，加快安装速度，节省空间

echo ">>> 项目根目录: $PROJECT_ROOT"
echo ">>> Conda 环境名: $ENV_NAME"
echo ">>> environment.yml: $ENV_YML"
echo ">>> VERL 目录: $VERL_DIR"
echo ">>> 训练脚本: $RUN_SCRIPT"
echo


### ==== 0️⃣ Conda 初始化（关键）====
# 若 conda 不在当前 shell 中可用，尝试自动加载初始化脚本
if ! command -v conda >/dev/null 2>&1; then
  echo "⚠️ conda 命令未加载，尝试自动初始化..."

  # 如果用户提前设置了 CONDA_HOME，则优先使用
  if [ -n "${CONDA_HOME:-}" ] && [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
    # 允许外部注入路径： export CONDA_HOME="$HOME/miniconda3"
    # shellcheck disable=SC1090
    source "$CONDA_HOME/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "❌ 未找到 conda.sh，请手动修改脚本中的路径，或先执行 'conda init bash' 并重新登录 shell。"
    exit 1
  fi
else
  # conda 已在 PATH 中，但有时仍需 hook（在非交互 shell 中）
  # 优雅方式：eval "$(conda shell.bash hook)"。若失败则继续使用当前会话。
  if conda shell.bash hook >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  fi
fi

# 再次确认
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ 仍未检测到 conda 命令。"
  exit 1
fi
echo "✅ conda 已可用。"
echo

# ---------- 1️⃣ 检查环境是否存在 ----------
if conda env list | grep -q "${ENV_NAME}"; then
    echo "✅ 检测到环境 ${ENV_NAME} 已存在，跳过创建与安装步骤。"
else
    echo "🚀 环境 ${ENV_NAME} 不存在，开始创建并安装依赖..."
    conda env create -f "${ENV_YML}"

    echo ">>> 激活环境 ${ENV_NAME}"
    conda activate "${ENV_NAME}"

    echo ">>> 安装 verl (editable mode + vllm,gpu)"
    cd "${VERL_DIR}"
    pip install --no-deps -e .[vllm,gpu]
fi

# ---------- 2️⃣ 激活环境 ----------
echo ">>> 激活环境 ${ENV_NAME}"
conda activate "${ENV_NAME}"
echo "✅ 成功激活环境 ${ENV_NAME}"

### ==== 3️⃣ 执行训练脚本 ====
echo ">>> 开始执行训练脚本：$RUN_SCRIPT"
cd "$VERL_DIR"   # 可根据需求切换工作目录
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

# 需要打包的目录（按存在性过滤）
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

  # 打包
  tar -czf "${ARCHIVE}" "${TO_PACK[@]}"
  SIZE_MB="$(du -m "${ARCHIVE}" | awk '{print $1}')"
  echo "✅ 生成压缩包：${ARCHIVE} (${SIZE_MB} MB)"

  # 若超过 45MB，自动分卷
  if [ "${SIZE_MB}" -gt 45 ]; then
    echo "📦 压缩包 > 45MB，自动分卷以便 Overleaf 上传..."
    split -b 45M -d -a 2 "${ARCHIVE}" "${ARCHIVE}.part-"
    echo "🧩 分卷完成：$(ls -1 ${ARCHIVE}.part-* | wc -l) 个分卷"
    echo "ℹ️ 如需在本地合并：cat ${ARCHIVE}.part-* > ${ARCHIVE}"
  fi

  echo "🎯 打包完成，导出目录：${EXPORT_DIR}"
fi