# A Semi-Decoupled Approach to Fast and Optimal Hardware-Software Co-Design of Neural Accelerators

[A Semi-Decoupled Approach to Fast and Optimal Hardware-Software Co-Design of Neural Accelerators](https://arxiv.org/)

Bingqian Lu, Zheyu Yan, Yiyu Shi, [Shaolei Ren](https://intra.ece.ucr.edu/~sren/), Proceedings of tinyML Research Symposium, Mar, 2022. (**tinyML’22**)

```BibTex
@article{
  luSemiCodesign2022,
  title={A Semi-Decoupled Approach to Fast and Optimal Hardware-Software Co-Design of Neural Accelerators},
  author={Bingqian Lu and Zheyu Yan and Yiyu Shi and Shaolei Ren},
  journal = {Proceedings of tinyML Research Symposium}, 
  month = Mar,
  year = 2022, 
  numpages = {7},
}
```

# Installation

This repository is based on [MAESTRO](https://maestro.ece.gatech.edu/docs/build/html/index.html), so the installation steps are the same as MAESTRO.
First clone this repository (which is forked from [MAESTRO](https://github.com/maestro-project/maestro)):
```
git clone https://github.com/Ren-Research/maestro.git
```

Then go the the downloaded directory:
```
cd maestro
```

Next install prerequisite packages on your machine if they are not previously installed (otherwise feel free to skip this step)
```
sudo apt install g++
sudo apt install build-essential
apt-get install scons
sudo apt-get install -y libboost-all-dev
```

Then compile the code
```
scons
```

# Running Experiments

Since MAESTRO originally does not support Pytorch models, we first modify the source code in ```maestro/tools/frontend/frameworks_to_modelfile_maestro.py``` to enable simulation of Pytorch models.

We consider NAS-Bench-301 and AlphaNet search sapces. The source code of sampling models and related data files used in our experiments for two search spaces are in directories ```maestro/tools/frontend/nasbench/``` and ```maestro/tools/frontend/alpha/``` respectively.

## NAS-Bench-301

We first sample 10k models. Then, based on the accuracy given by NAS-Bench-301 and FLOPs of these 10K models, we select 1017 models, including the Pareto-optimal front (in terms of predicted accuracy and FLOPs) and some random architectures.

In our experiment, 10k sampled model configs are in ```maestro/tools/frontend/nasbench/new/model_arch_depth.pickle```, 1017 selected model indexes are in ```maestro/tools/frontend/nasbench/new/model_index_depth.pickle```.

We then generate MAESTRO DNN model files for these 1017 models:

```
python frameworks_to_modelfile_maestro.py --api_name pytorch --custom 111 --model custom --outfile dnn_model.m --input_size 3,32,32
```

The generated MAESTRO DNN model files will be stored in ```maestro/data/model/```, in the file names of ```0_dnn_model.m```, ```1_dnn_model.m```, etc.


Next, we generate MAESTRO mapping files with the MAESTRO DNN model file and specific dataflow:

```
python modelfile_to_mapping.py --dataflow 'rs'
```

The generated MAESTRO mapping files will be stored in ```maestro/data/mapping/```, in the file names of ```0_out.m```, ```1_out.m```, etc.


The final step is to run MAESTRO simulation with the generated mapping:

```
cd ../..
python execute_run_example.py --dataflow 'rs'
```

The generated MAESTRO simulation analysis results for each model, specific dataflow and given hardware config will be ```maestro/```, in the file names of ```0_out.csv```, ```1_out.csv```, etc.


## AlphaNet

We first sample 10k models and then select 1046 models based on the predicted accuracy given by the released accuracy predictor and FLOPs.

In our experiment, 10k sampled model configs are in ```maestro/tools/frontend/alpha/model_arch_depth.pickle```, 1046 selected model indexes are in ```maestro/tools/frontend/alpha/model_index_depth.pickle```.

We then generate MAESTRO DNN model files for these 1046 models:

```
python frameworks_to_modelfile_maestro.py --api_name pytorch --custom 111 --model custom --outfile dnn_model.m --input_size 3,32,32
```

The generated MAESTRO DNN model files will be stored in ```maestro/data/model/```, in the file names of ```0_dnn_model.m```, ```1_dnn_model.m```, etc.


Next, we generate MAESTRO mapping files with the MAESTRO DNN model file and specific dataflow:

```
python modelfile_to_mapping.py --dataflow 'rs'
```

The generated MAESTRO mapping files will be stored in ```maestro/data/mapping/```, in the file names of ```0_out.m```, ```1_out.m```, etc.


The final step is to run MAESTRO simulation with the generated mapping:

```
cd ../..
python execute_run_example.py --dataflow 'rs'
```

The generated MAESTRO simulation analysis results for each model, specific dataflow and given hardware config will be ```maestro/```, in the file names of ```0_out.csv```, ```1_out.csv```, etc.


Below is the original readme of MAESTRO.

====================================================================================================

# MAESTRO: An Open-source Infrastructure for Modeling Dataflows within Deep Learning Accelerators
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

# What is MAESTRO?
MAESTRO is an open-source tool for modeling and evaluating the performance and energy-efficiency of different dataflows. MAESTRO is actively developed by the [Synergy Lab](https://synergy.ece.gatech.edu/) at [Georgia Institute of Technology](https://www.gatech.edu/). For more details about MAESTRO, please visit the following links.

- [MAESTRO Website](http://maestro.ece.gatech.edu/)
- [MAESTRO Docs](http://maestro.ece.gatech.edu/docs/build/html/index.html)


# Codebase

## Updates
### May 26th, 2021

We updated the hardware description file, added off-chip bandwidth added as constraint.

We added a validation folder with data for Eyeriss and MAERI from MICRO 2019 paper.

### Oct 13th, 2020

We added a direct support for GEMM layers. For more information, please take a look at [here](http://maestro.ece.gatech.edu/docs/build/html/layer_supported.html).

### May 13th, 2020

We updated the naming convention of mappings and the directory structure of data folder.

### Oct 14th, 2019

Latest codebase released along with MAESTRO MICRO 2019 paper.


## Maintainers
- Felix (Sheng-Chun) Kao (felix@gatech.edu)
- Geonhwa Jeong (geonhwa.jeong@gatech.edu)
- Tushar Krishna (tushar@ece.gatech.edu)


## Technical Contributors
- Hyoukjun Kwon (Georgia Tech, now at Facebook Reality Labs): Main developer (core framework and functionalities)
- Prasanth Chatarasi (Georgia Tech, now at IBM Research): APIs + interface to mapping optimizers.
- Felix (Sheng-Chun) Kao (Georgia Tech): Pytorch frontend + updates to cost-model/interface + GAMMA mapper
- Geonhwa Jeong (Georgia Tech): Keras frontend + debugging + website maintainer.
- Saurabh Malik (Georgia Tech, now at Microsoft): Jupyter Notebooks demo + website.

# Citations ###
```
@inproceedings{maestro_micro2019,
  author    = {Hyoukjun Kwon and
               Prasanth Chatarasi and
               Michael Pellauer and
               Angshuman Parashar and
               Vivek Sarkar and
               Tushar Krishna},
  title     = {Understanding Reuse, Performance, and Hardware Cost of {DNN} Dataflow:
               {A} Data-Centric Approach},
  booktitle = {Proceedings of the 52nd Annual {IEEE/ACM} International Symposium
               on Microarchitecture, {MICRO}},
  pages     = {754--768},
  publisher = {{ACM}},
  year      = {2019},
}

```
```
@article{maestro_toppicks2020,
  author    = {Hyoukjun Kwon and
               Prasanth Chatarasi and
               Vivek Sarkar and
               Tushar Krishna and
               Michael Pellauer and
               Angshuman Parashar},
  title     = {{MAESTRO:} {A} Data-Centric Approach to Understand Reuse, Performance,
               and Hardware Cost of {DNN} Mappings},
  journal   = {{IEEE} Micro},
  volume    = {40},
  number    = {3},
  pages     = {20--29},
  year      = {2020},
}
```
