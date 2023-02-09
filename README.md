## LUCA

**conda**

```
# from https://docs.conda.io/en/latest/miniconda.html#linux-installers
curl -o ./Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_22.11.1-1-Linux-x86_64.sh
bash Miniconda3.sh
conda list
```

**CPU**

```
conda env create -f environment_cpu.yml
```

**CUDA**

_E.g._ on AWS as spot instance with `g4dn.xlarge`, `g4dn.2xlarge`, `g5.xlarge`, `g5.2xlarge` (see [G4 instances](https://aws.amazon.com/ec2/instance-types/g4/) and [G5 instances](https://aws.amazon.com/ec2/instance-types/g5/)) with a [Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/).

```
nvcc --version
nvidia-smi

conda env create -f environment_gpu.yml
```
