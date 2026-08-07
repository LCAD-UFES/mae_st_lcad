#!/bin/bash
set -e

# 1. Cria o ambiente
conda env create -f environment.yml

# 2. Carrega as funções do Conda no subshell do script
eval "$(conda shell.bash hook)"

# 3. Agora o activate funciona normalmente dentro do script
conda activate mae_st_lcad

# 4. Executa os pips no ambiente ativo
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install --no-build-isolation "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@b4a4a3bd136852dae5fb1de37978dee412653e31"