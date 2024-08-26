# Design and Implementation of a centralized Database for Managing Apache TVM Fine-Tuning Outputs across Heterogeneous Hardware Backends

## Introduction

This README provides an overview of the setup and configuration required to run the programs associated with this project, followed by a description of the Thesis Database application and brief explanations of each file. For supplementary information, please refer to the thesis PDF.

### System Specifications

- **Hardware**: Two Nvidia RTX 2080Ti (only one utilized)
- **Operating System**: Linux
- **Development Environment**: Jupyter Notebooks via SSH
- **Dataset**: IMAGENET

## Setup Instructions

### Prerequisites

Before beginning, ensure your system meets the following requirements:

- **Ubuntu**
- **Conda** installed
- **Python 3**

### TVM Installation Guide for Ubuntu

Follow these steps to install TVM on your system:

1. **Clone the TVM repository**:
    git clone --recursive https://github.com/apache/tvm tvm

2. **Download and install dependencies**:
    sudo apt-get update
    sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

3. **Set up the environment and create a build folder**:
    conda activate tvm-env
    cd tvm
    mkdir build
    cp cmake/config.cmake build
    cd build

4. **Edit the configuration**:
    nano config.cmake

5. **Run CMake**:
    echo $CONDA_PREFIX
    cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/conda/prefix

6. **Build and install**:
    make -j`nproc` install

7. **Install Python packages**:
    cd ../python
    python setup.py install 

8. **Install additional dependencies**:
    pip3 install --user numpy decorator attrs typing-extensions psutil scipy tornado psutil 'xgboost>=1.1.0' cloudpickle

9. **Set up Jupyter Notebook**:
    conda install -c conda-forge tensorflow anaconda ipykernel
    python -m ipykernel install --user --name=

10. **Run Jupyter Notebook**:
    jupyter notebook


## How the Program Works

The purpose of the database application is to create an automated system that captures and utilizes the best AutoTVM tuning configurations tailored to different ML models and hardware backends. Because the autotuning process can be time-consuming, saving the results of this process allows for a ready-to-use application that can apply these configurations without needing to repeat the autotuning process.

### Program Overview

The program is split into two main components:

1. **Tuning Program**: This program tunes the ML model to the specified characteristics such as batch size, backend, and model type. The tuning process can be initiated with the following command:
    python3 program1.py resnet-18 cuda 100

In this example:
resnet-18 is the model being tuned.
cuda specifies the backend (e.g., CUDA).
100 represents the batch size or other specific parameter.

2. **Execution Program**: This component runs the specified application based on the saved configurations. It is implemented as a Jupyter notebook to allow for easy visualization of the results, such as viewing images processed by the model.

### Applicability
While this application is specifically designed for computer vision models, it can be adapted for use with other types of models by making appropriate changes to the code.


### Key Points:
- **Clear Sections**: The content is divided into clear sections: "How the Program Works," "Program Overview," and "Applicability," making it easy to follow.
- **Example Command**: The example command is highlighted in a code block for easy identification.
- **Generalization**: The potential for adapting the program to other models is noted.

This structure ensures that users can easily understand the purpose of the application and how to use it effectively.

## Repository Files

This section highlights the most important files and their applications.

### tuning-logs
This directory contains the logs of various tuning runs, which are used to test different TVM tuning settings and batch configurations.

### tvm_report
This directory holds the core components of the application:

1. **automated_database.py**: Defines the path where the program saves the logged tuning results. It is essential for managing the database of tuning configurations.

2. **AutotuneCPU.py**: Contains the autotuning code specifically for CPUs. This file was used prior to the merging of CPU and GPU autotuning into a unified approach.

3. **AutotuneGPU.py**: Contains the autotuning code specifically for GPUs. This file was used before the CPU and GPU autotuning functionalities were combined.

4. **Bart_Batches.ipynb**: Demonstrates the use of the BART model for natural language processing with TVM. This notebook shows that TVM supports not only computer vision models but also NLP tasks.

5. **Run_tvm.ipynb**: A Jupyter Notebook used to execute the TVM database application. This notebook helps in running and managing TVM-based experiments and evaluations.

6. **ViT_batches.ipynb**: Focuses on running a Vision Transformer (ViT) model for complex image classification tasks. This notebook illustrates TVM's capability to handle advanced image classification models.

7. **extract_tune.ipynb**: A Jupyter Notebook designed to extract and analyze information related to tuning tasks. It provides insights before and during the autotuning process managed by AutoTVM.

8. **program1.py**: The executable Python script for the database application. It serves as the main entry point for interacting with the tuning database.

Feel free to explore each file to understand their roles and how they contribute to the overall project.

 

## Appendix

### TVM File Management

During the course of experimentation, significant challenges arose concerning the management of TVM's temporary files, which are stored by default in `/tmp`. In order to gain clearer insights into the functionality and impact of these files, approximately 9GB of accumulated temporary data on the machine were systematically cleared. This process was crucial for enabling the re-execution of TVM's task extraction (`tvm-extract-tasks`), thereby allowing for a deeper understanding of each file's role.

Upon analysis, the process revealed the following workflow: the builder generated a `tvm_tmpzz4_s6p/kernels.cu` file, which subsequently compiled into an executable CUDA file. This file was then packaged into a tar archive located within a randomly generated 64-bit number directory (e.g., `/tmp/tmpuy0f0_5r/tmp_func_dba47db8cbef2ce0.tar`), containing compiled objects (`devc.o` and `lib0.o`). These archives were passed to the runner for execution.

Further investigation indicated that each tuning task and configuration space initiated the creation of a new tar archive. This systematic approach ensured that each task and configuration were appropriately encapsulated for execution within TVM's framework.
