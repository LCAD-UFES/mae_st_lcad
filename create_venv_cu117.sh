#!/bin/bash
set -e

# 1. Cria o ambiente
conda env create -f environment.yml

# 2. Carrega as funções do Conda no subshell do script
eval "$(conda shell.bash hook)"

# 3. Agora o activate funciona normalmente dentro do script
conda activate mae_st_lcad

# 4. Executa os pips no ambiente ativo
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install "setuptools<70.0.0" wheel
pip install --no-build-isolation "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@b4a4a3bd136852dae5fb1de37978dee412653e31"
pip install opencv-python
pip install "numpy<2" --force-reinstall
pip install psutil
mkdir checkpoints
mkdir results
wget 'https://dl.fbaipublicfiles.com/video-mae-200x4-nonorm.pth' -O checkpoints/video-mae-200x4-nonorm.pth