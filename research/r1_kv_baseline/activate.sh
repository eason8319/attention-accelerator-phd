# 集群环境激活脚本（source 使用，不要直接执行）
#   source research/r1_kv_baseline/activate.sh
#
# 背景与踩坑细节见本地文件 CLUSTER_NOTES.md（不入库）。

source /cluster/apps/miniconda3/etc/profile.d/conda.sh
conda activate r1-kv-baseline

export HF_HOME="${HOME}/hf-cache"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

# huggingface.co 从本集群不可达，改用镜像（详见 CLUSTER_NOTES.md）。
export HF_ENDPOINT="https://hf-mirror.com"

echo "[r1-kv-baseline] python: $(python -V 2>&1) | torch cuda build: $(python -c 'import torch;print(torch.version.cuda)' 2>/dev/null)"
echo "[r1-kv-baseline] HF_HOME=$HF_HOME | HF_ENDPOINT=$HF_ENDPOINT"
echo "[r1-kv-baseline] 提示：登录节点无 GPU；需要 GPU 时用 srun/sbatch 提交到 'gpu' 分区，例如："
echo "    srun -p gpu --gres=gpu:1 --pty bash -c 'source research/r1_kv_baseline/activate.sh && python -c \"import torch; print(torch.cuda.is_available())\"'"
