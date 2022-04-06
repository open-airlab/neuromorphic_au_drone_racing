#!/bin/bash

echo Installing requirements
pip install -r requirements.txt

echo Cloning rpg_asynet
git clone https://github.com/uzh-rpg/rpg_asynet.git

echo Cloning SparseConvNet
git clone https://github.com/facebookresearch/SparseConvNet.git

echo Installing pytorch through conda
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch 

echo Installing SparseConvNet
cd SparseConvNet
bash develop.sh

echo Cloning Eigen
cd ..
git clone https://gitlab.com/libeigen/eigen.git rpg_asynet/async_sparse_py/include/eigen

echo Installing event_representation_tool
cd rpg_asynet
pip install event_representation_tool/

echo Installing pybind11
conda install -c conda-forge pybind11

echo Installing async_sparse_py
pip install async_sparse_py/