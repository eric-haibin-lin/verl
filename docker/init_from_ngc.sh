pip3 uninstall xgboost transformer_engine flash_attn -y

# make sure torch version is kept
pip3 install --no-cache-dir \
    "torch<=2.4.0a0+3bcc3cddb5.nv24.7" \
    accelerate \
    codetiming \
    dill \
    hydra-core \
    numpy \
    pybind11 \
    tensordict \
    "transformers<=4.46.0"

# ray is installed via vllm
pip3 install --no-cache-dir vllm==0.6.3

pip3 install --no-cache-dir --no-build-isolation flash-attn==2.7.0.post2



apt-get update && apt-get install vim -y

pip3 uninstall torch torchvision torchaudio -y

# make sure torch version is kept
pip3 install --no-cache-dir \
    "torch==2.4.0" \
    accelerate \
    codetiming \
    dill \
    hydra-core \
    numpy \
    pybind11 \
    tensordict \
    "transformers<=4.46.0"

# ray is installed via vllm
pip3 install --no-cache-dir vllm==0.6.3

pip3 install --no-cache-dir --no-build-isolation flash-attn==2.7.0.post2




# uninstall nv-pytorch fork
pip3 uninstall pytorch-quantization \
     pytorch-triton \
     torch \
     torch-tensorrt \
     torchvision \
     xgboost transformer_engine flash_attn \
     apex megatron-core -y

pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# make sure torch version is kept
pip3 install --no-cache-dir \
    "torch==2.4.0" \
    accelerate \
    codetiming \
    datasets \
    dill \
    hydra-core \
    numpy \
    pybind11 \
    tensordict \
    "transformers<=4.46.0"

# ray is installed via vllm
pip3 install --no-cache-dir vllm==0.6.3

# we choose flash-attn v2.7.0 or v2.7.2 which contain pre-built wheels
pip3 install --no-cache-dir --no-build-isolation flash-attn==2.7.0.post2

git clone https://github.com/volcengine/verl
cd verl
pip3 install -e .


