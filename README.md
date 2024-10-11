Instructions for replicating envirnonment used. (Linux Host system)

Install conda if not installed. 

$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$ export PATH="$HOME/miniconda3/bin:$PATH"
$ source ~/.bashrc
$ conda --version
# conda 24.5.0

Create conda envirnonment. 

$ conda create --name LLAMA python=3.12
$ conda init (if needed)
Reopen shell. 
$ conda activate LLAMA 

$ python --version
Python 3.12.7

$ pip install these modules:
$ pip install huggingface

tranformers

$ conda install these modules (faster):
$ conda install pytorch cudatoolkit -c pytorch

$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jun_14_16:34:21_PDT_2024
Cuda compilation tools, release 12.6, V12.6.20
Build cuda_12.6.r12.6/compiler.34431801_0

Make sure cuda is detected in system. Use following python script to verify:

import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

Update CUDA to PATH if needed:

$ export PATH=/usr/local/cuda/bin:$PATH
$ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
$ source ~/.bashrc

