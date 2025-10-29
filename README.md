# MT_Grpo — 使用说明

## 0）前置条件

- 已安装 **Conda**、**Git**、**NVIDIA 驱动**（可执行 nvidia-smi）。
- 上传结果文件前，加入 Overleaf 项目 **“XJF_Logs”**。





## 1）项目运行步骤

登陆服务器，终端运行：

```shell
# 1）创建工作目录
mkdir -p /mnt/workspace/xintong/xjf
cd /mnt/workspace/xintong/xjf

# 2）从 GitHub 下载项目
git clone https://github.com/jetfan-xin/MT_Grpo.git
cd MT_Grpo/scripts

# 3）启动一个新的 tmux 会话
tmux new -s xjf_mtgrpo

# 4）运行主脚本
chmod +x setup_and_run.sh
bash setup_and_run.sh
```

脚本自动执行：

- 初始化 Conda
- 创建环境 xjf_verl（Python 3.10）
- 检测 CUDA 并安装匹配版本的 PyTorch
- 安装依赖与 flash-attn
- 模型训练+推理 verl/custom_grpo_fast_qe.sh
- 打包结果至 exports/MT_Grpo_logs_*.tar.gz（大于45MB自动分卷）


```shell
# 5）完成后退出会话：
exit
# 6）或者从外部关闭
# tmux kill-session -t mtgrpo
```

### 排错

` setup_and _run.sh` 脚本出错时，可上传同目录下（MT_Grpo/scripts）的 `setup_and_run.sh` 文件方便排错。




## 2）结果文件

输出位置：`/mnt/workspace/xintong/xjf/MT_Grpo/exports/`

其中包含：`MT_Grpo_logs_{运行_时间}.tar.gz`或者分卷文件 `*.part-00, *.part-01...` 





## 3）上传结果到 Overleaf

1. 受邀加入并打开 Overleaf 项目 **“XJF_Logs”**
2. 进入文件夹 MT_Grpo/
3. 上传生成的 MT_Grpo_logs_\*.tar.gz 或所有 .part-\* 分卷文件