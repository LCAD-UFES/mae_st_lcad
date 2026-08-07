Navegação rápida

- [Installing CUDA 12.8 (Versão compatível com a RTX 5070 e todas as outras)](#instalando-cuda-128-versão-atual-e-compatível-com-o-mot)
  - [ATENÇÃO (Antes de começar)](#atenção-antes-de-começar)
  - [Instalando drivers da placa de vídeo](#instalando-drivers-da-placa-de-vídeo)
  - [Instalação de CUDA 12.8 (Ubuntu 20.04.5)](#instalação-do-cuda-128)
  - [Instalar o CuDNN](#instalar-o-cudnn)
  - [Instalar o TensorRT](#instalar-o-tensorrt)
  - [Instalar o Python3.10](#instalar-o-python310-obrigatório)
# Instalando CUDA 12.8 (Ubuntu 20.04)

## ATENÇÃO (Antes de começar)

#### Se já alguma etapa dessas abaixo já estiver sido feita será necessário remover os drivers e/ou a versão do CUDA instalada, para isso siga os passos abaixo ou prossiga caso seja a primeira configuração da máquina:


Execute os comandos abaixo em um terminal:

```
sudo apt --purge remove "nvidia*" "libnvinfer*" "libcudnn*" "tensorrt*" "cuda*"
sudo apt autoremove
sudo rm -rf /usr/local/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
cp ~/.bashrc ~/.bashrc.backup
sed -i '/cuda/d' ~/.bashrc
```

Após a remoção reinicie a maquina e depois prossiga com as próximas instalações

## Adicione o repositório que contém os drivers mais novos da nvidia

Adicionando o repositório:
```
sudo add-apt-repository ppa:graphics-drivers/ppa
```

```
sudo apt-get update
```
Confira se o nvidia-driver-580 está disponível:
```
apt-cache search nvidia-driver-580
```
Caso não seja listado, faça uma limpeza no cache do APT executando os seguintes comandos:

```
sudo apt clean
```

```
sudo rm -rf /var/lib/apt/lists/*
```

```
sudo apt update
```


## Instalando drivers da placa de vídeo

#### Série RTX 3000/4000:

```
sudo apt-get install nvidia-driver-580

```


#### Série RTX 5000:

```
sudo apt-get install nvidia-driver-580-open

```

Após a instalação reinicie a maquina e depois prossiga com as próximas instalações

## Instalação do CUDA 12.8

Executa os comandos abaixo para o download e instalação:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2004-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH'  >> ~/.bashrc
. ~/.bashrc
```

Após a instalação reinicie a maquina e depois prossiga com as próximas instalações


## Instalar o CuDNN

Acesse [NVIDIA CuDNN](https://developer.nvidia.com/rdp/cudnn-archive), e clique em Download CuDNN. Na página seguinte, crie uma conta ou acesse a sua, caso já tenha criado.

Na página CuDNN Download, marque a caixa **I Agree To the Terms of the cuDNN Software License Agreement** e no fim da página clique em Clique em **Download cuDNN v8.9.7 (December 5th, 2023), for CUDA 12.x** e baixe o seguinte arquivo:
- Local Installer for Ubuntu20.04 x86_64 (Deb)

Abra o terminal na pasta dos downloads e instale o pacote:
```
cd ~/Downloads
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.9.7.29_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-8.9.7.29/cudnn-local-30472A84-keyring.gpg /usr/share/keyrings/
sudo apt-get update
cd /var/cudnn-local-repo-ubuntu2004-8.9.7.29/
sudo dpkg -i libcudnn8_8.9.7.29-1+cuda12.2_amd64.deb
sudo dpkg -i libcudnn8-dev_8.9.7.29-1+cuda12.2_amd64.deb
sudo dpkg -i libcudnn8-samples_8.9.7.29-1+cuda12.2_amd64.deb
```
Para testar:
```
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd ~/cudnn_samples_v8/mnistCUDNN/
sed -i 's/\$(HIGHEST_SM)/86/g; s/\$(sm)/86/g' Makefile
make clean && make
./mnistCUDNN
```
Caso um dos seguintes erros aconteçam:
```
test.c:1:10: fatal error: FreeImage.h: No such file or directory
    1 | #include "FreeImage.h"
      |          ^~~~~~~~~~~~~
compilation terminated.
```
Faça:
```
sudo apt-get install -y libfreeimage-dev
```
Se for este:
```
error while loading shared libraries: libcudart.so.9.0: cannot open shared object file: No such file or directory
```
Faça:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```
Após a instalação reinicie a maquina e depois prossiga com a última instalação

## Instalar o TensorRT

Download Tensorrt: 
Link: https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.10.0/local_repo/nv-tensorrt-local-repo-ubuntu2004-10.10.0-cuda-12.9_1.0-1_amd64.deb


Agora abra o terminal na pasta de downloads e execute os seguintes comandos para instalar o TensorRT:
```
cd ~/Downloads
sudo dpkg -i $HOME/Downloads/nv-tensorrt-local-repo-ubuntu2004-10.10.0-cuda-12.9_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2004-10.10.0-cuda-12.9/nv-tensorrt-local-51C388DF-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo rm $HOME/Downloads/nv-tensorrt-local-repo-ubuntu2004-10.10.0-cuda-12.9_1.0-1_amd64.deb
sudo apt install tensorrt
```

## Instalar o Python3.10 (Obrigatório):

### Baixe e descompacte o pacote de instalação [Python-3.10.18](https://drive.google.com/file/d/1j01p2rh4GAN5wLueZrhWSXfEZnCnfvSM/view?usp=sharing):

```
mv ~/Downloads/python310.tar.gz ~/packages_astro
cd ~/packages_astro
tar -xf python310.tar.gz
```
### Agora instale os pacotes necessários:

```
sudo apt install libjs-mathjax
cd python310
sudo apt install  ./*.deb
```
### Se houver problemas faça:
```
sudo apt update
sudo apt --fix-broken install
sudo dpkg --configure -a
sudo dpkg -i *.deb
```

### Verifique a instalação:
```
python3.10 --version
```