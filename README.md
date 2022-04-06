# neuromorphic_au_drone_racing
This repository contains the public datasets and code for event camera perception for drone racing. The dataset is comprised of two parts: N-AU-DR-sim containing simulated data and N-AU-DR-real containing real event data.

Paper: [link](https://arxiv.org/abs/2204.02120)

To cite this paper:

```
  @INPROCEEDINGS{kristoffer2022ecc,
  author={Andersen, Kristoffer Fogh and Pham, Huy Xuan and Ugurlu, Halil Ibrahim and Kayacan, Erdal},
  booktitle={2022 European Control Conference (ECC)}, 
  title={Event-based Navigation for Autonomous Drone Racing with Sparse Gated Recurrent Network}, 
  year={2022}}
```

## Download the dataset
- The entire dataset can be downloaded from: [N-AU-DR](https://drive.google.com/drive/folders/1InoQCxsV5SEjSLCI9hsjjMPBX8-QMu_H?usp=sharing)
(10GB compressed and 33GB uncompressed). 

- Each dataset is split into a training set and a validation set.

## Dataset description
- The overall structure of the dataset is aligned with how the [Gen1 Automotive dataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) is structured, and all tools compatible with this dataset can also be used for our dataset. 
- The dataset is organized into .dat files each containing 60 seconds of event data, either simulated or real. Each dat file is a binary file in which events are encoded using 4 bytes (unsigned int32) for the timestamps and 4 bytes (unsigned int32) for the data
- The data is composed of 14 bits for the x position, 14 bits for the y position and 1 bit for the polarity (encoded as -1/1).

- For N-AU-DR-sim, the .dat filenames identify the date, timestamp and the simulated contrast threshold used. For example: 29-04-2021_15-21-46_1619702295_0.3_td.dat. 

- For N-AU-DR-real train split, the .dat filenames identify the date, timestamp, number of drone rotations and drone velocity. For example: 18-06-2021_10-57-20_1624006640_6rot_075ms_td.dat

- For N-AU-DR-real validation split, the .dat filenames identify the date, timestamp, environment lighting and whether gates have the same placement as in the training data. For example: 10-07-2021_13-58-01_1625918281_50light_gatemoved_td.dat

- For each .dat file there is a corresponding .npy file containing the annotated bounding boxes of gates. Each bounding box consist of `x` abscissa of the top left corner in pixels, `y` ordinate of the top left corner in pixels, `w` width of the boxes in pixel, `h` height of the boxes in pixel, `ts` timestamp of the box in the sequence in microseconds, `class_id` 0 for gates.


## Dataset usage
For visualizing and manipulating the dataset we recommend using the [prophesee-automotive-dataset-toolbox](https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox). Clone this repository and it then provides a variety of tools for visualizing the data with annotations.

## Event-based gate detector with ROS

### Requirements
- Tested on ROS Melodic + Ubuntu 18.04
- Event camera running ros driver rpg_dvs_ros: https://github.com/uzh-rpg/rpg_dvs_ros
### Trained models
- We provided 3 trained models: our sparse_RNN and two baselines dense_VGG and sparse_VGG under the following [link](https://drive.google.com/drive/folders/17RMxHaaM6bpOd4dzEvmqZNyxy03sBUe0?usp=sharing)
- Download the respective model and put them into folder "trained_models" (currently empty)
### Installation
    
- Executing develop.sh will install all dependencies and external libraries (including Facebook's SparseConvNet):

```
cd ros/
./develop.sh
```

### Usage
- Launch package "event_preprocessor" for creating event histogram:

```
roslaunch event_preprocessor event_preprocessor.launch
```
- Then, launch detector: 
```
roslaunch event_detector_rt detector.launch
```
