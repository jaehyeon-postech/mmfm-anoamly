export CUDA_HOME=/data/02/jiacong/cuda-12.1

conda create -n anomaly_ov3_7b python=3.10 -y
conda activate anomaly_ov3_7b
pip install --upgrade pip
pip install -e ".[train]"

pip install accelerate==0.29.3

pip install flash-attn --no-build-isolation

pip install pynvml