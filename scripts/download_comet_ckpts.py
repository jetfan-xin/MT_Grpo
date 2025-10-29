#!/usr/bin/env python3
"""
download_comet_ckpts.py

用法：
    python3 download_comet_ckpts.py --comet_dir /path/to/wmt23-cometkiwi-da-xl --word_qe_dir /path/to/XCOMET-XL

功能：
    - 检查 COMET 与 XCOMET 的 checkpoints 是否已存在
    - 若不存在则从 Hugging Face 下载到指定目录（扁平结构）
    - 若存在则直接跳过
"""

import os
import argparse
from huggingface_hub import snapshot_download

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ensure_model(model_id: str, target_dir: str):
    """下载指定模型到目标目录（包含 checkpoints/model.ckpt）"""
    ckpt_path = os.path.join(target_dir, "checkpoints", "model.ckpt")
    if os.path.exists(ckpt_path):
        print(f"✅ 模型已存在：{ckpt_path}")
        return ckpt_path

    print(f"⬇️  开始下载 {model_id} 到 {target_dir} ...")

    ensure_dir(os.path.join(target_dir, "checkpoints"))

    snapshot_download(
        repo_id=model_id,
        allow_patterns=["checkpoints/*", "hparams.yaml", "LICENSE", "README.md"],
        local_dir=target_dir,                 # 下载到目标目录
        local_dir_use_symlinks=False,         # 不使用软链接，直接复制文件
        resume_download=True,                 # 支持断点续传
    )

    if os.path.exists(ckpt_path):
        print(f"✅ 下载完成：{ckpt_path}")
    else:
        print(f"⚠️ 下载完成但未检测到 {ckpt_path}，请检查下载内容：{target_dir}")
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Download COMET & XCOMET checkpoints to fixed directories.")
    parser.add_argument("--comet_dir", type=str, required=True, help="目标目录，用于存放 COMET (wmt23-cometkiwi-da-xl)")
    parser.add_argument("--word_qe_dir", type=str, required=True, help="目标目录，用于存放 XCOMET-XL")
    args = parser.parse_args()

    ensure_dir(args.comet_dir)
    ensure_dir(args.word_qe_dir)

    comet_ckpt = ensure_model("Unbabel/wmt23-cometkiwi-da-xl", args.comet_dir)
    word_qe_ckpt = ensure_model("Unbabel/XCOMET-XL", args.word_qe_dir)

    print("\n🎉 COMET & XCOMET 模型已准备完成。")
    print(f"COMET_CKPT={comet_ckpt}")
    print(f"WORD_QE_CKPT={word_qe_ckpt}")


if __name__ == "__main__":
    main()