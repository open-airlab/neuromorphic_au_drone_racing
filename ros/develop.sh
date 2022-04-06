#!/bin/bash

echo Installing requirements
pip install -r requirements.txt

echo Cloning SparseConvNet
cd event_detector_rt/nodes/models/
git clone https://github.com/facebookresearch/SparseConvNet.git

echo Installing pytorch through conda
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch 

echo Installing SparseConvNet
cd event_detector_rt/nodes/models/SparseConvNet
bash develop.sh


