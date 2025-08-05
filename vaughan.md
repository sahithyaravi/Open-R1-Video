# Installation Instructions for the Cluster

Follow the steps below to set up the environment and install the required dependencies:

- You need to install this version of transformers first from video-r1 repo and unzip and install: 

## 1. Load Required Modules
```bash
module load python/3.10.13 
module load scipy-stack/2025a
module load gcc opencv
module load gcc arrow/19.0.1
module load gcc cuda
module load rust
```

## 2. Set up a Virtual Environment
```bash
conda create -n videor1 python=3.10
```



## 3. Activate the Virtual Environment
```bash
conda activate videor1
```

## 4. Install `av` Package
```bash
conda install av -c conda-forge
conda install cmake
```
## 5. Install Additional Dependencies
```bash
cd ..
pip3 install -e ".[dev]"
pip3 install flash_attn --no-build-isolation
```

## 6. Install `qwen-vl-utils`
```bash
cd qwen-vl-utils
pip install -e .
```

## 7. Install transformers from Video R1
```bash
pip uninstall transformers
```

Download "https://drive.google.com/file/d/1Kc81WZitEhUZYWXpL6y2GXuSXufLSYcF/view".
```bash
unzip transformers-main.zip
cd ./transformers-main
pip install .
pip install trl==0.16.0
pip install numpy==1.22.4
```
You are now ready to use the cluster environment!  