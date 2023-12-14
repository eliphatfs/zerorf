# ZeroRF: Sparse View 360° Reconstruction with Zero Pretraining

## Requirements

As the code is based on [the SSDNeRF codebase](https://github.com/Lakonik/SSDNeRF), the requirements are the same. Additionally, we provide a docker image for ease of use.

### Install via Docker

```bash
docker pull eliphatfs/zerorf-ssdnerf:0.0.2
```

### Install Manually (Copied from SSDNeRF)

The code has been tested in the environment described as follows:

- Linux (tested on Ubuntu 18.04/20.04 LTS)
- Python 3.7
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 11
- [PyTorch](https://pytorch.org/get-started/previous-versions/) 1.12.1
- [MMCV](https://github.com/open-mmlab/mmcv) 1.6.0
- [MMGeneration](https://github.com/open-mmlab/mmgeneration) 0.7.2
- [SpConv](https://github.com/traveller59/spconv) 2.3.6

Other dependencies can be installed via `pip install -r requirements.txt`. 

An example of commands for installing the Python packages is shown below (you may change the CUDA version yourself):

```bash
# Export the PATH of CUDA toolkit
export PATH=/usr/local/cuda-11.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64:$LD_LIBRARY_PATH

# Create conda environment
conda create -y -n ssdnerf python=3.7
conda activate ssdnerf

# Install PyTorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Install MMCV and MMGeneration
pip install -U openmim
mim install mmcv-full==1.6
git clone https://github.com/open-mmlab/mmgeneration && cd mmgeneration && git checkout v0.7.2
pip install -v -e .
cd ..

# Install SpConv
pip install spconv-cu114

# Clone this repo and install other dependencies
git clone <this repo> && cd <repo folder> && git checkout ssdnerf-sd
pip install -r requirements.txt
```

Optionally, you can install [xFormers](https://github.com/facebookresearch/xformers) for efficnt attention. Also, this codebase should be able to work on Windows systems as well (tested in the inference mode).

Lastly, there are two CUDA packages from [torch-ngp](https://github.com/ashawkey/torch-ngp) that need to be built locally if you install dependencies manually.

```bash
cd lib/ops/raymarching/
pip install -e .
cd ../shencoder/
pip install -e .
cd ../../..
```

## Running
