## LUCA

Using [MagicSoup](https://pypi.org/project/magicsoup/) to simulate cell evolution.
I organized experiments in directories that each have preparation, simulation, and analysis in it.

- [e0_co2_fixing](./e0_co2_fixing) bring cells to fix CO2

## CPU-only Setup

For CPU-only with [conda](https://docs.conda.io/en/latest/) it is:

```
conda env create -f environment_cpu.yml
```

## AWS EC2 with GPU

Use one of the [Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/) with an instance like
`g4dn.xlarge`, `g4dn.2xlarge`, `g5.xlarge`, `g5.2xlarge` (see [G4 instances](https://aws.amazon.com/ec2/instance-types/g4/) and [G5 instances](https://aws.amazon.com/ec2/instance-types/g5/)).
_E.g._ **Deep Learning AMI GPU PyTorch 1.12.1 (Ubuntu 20.04) 20221114** (ami-01e8ee929409916a3) has CUDA 11 and conda installed.
After starting the instance you can initialize conda and directly install the environment.

```bash
git clone https://github.com/mRcSchwering/luca  # get repo

conda init && source ~/.bashrc  # init conda
conda env create -f environment_gpu.yml  # install environment
conda activate luca  # activate envrionment

nvcc --version  # check CUDA version
python -c 'import torch; print(torch.cuda.is_available())'  # check torch was compiled for it
nvidia-smi -l 1  # monitor GPU
```

## Screen

A little cheatsheet for [Screen](https://wiki.ubuntuusers.de/Screen/)
because I always forget the commands.

- `screen -ls` list active sessions
- `screen -S asd` start new session _asd_
- `screen -r asd` return to active session _asd_
- `screen -XS asd quit` kill session _asd_
- when attached to session **Ctrl + A** then **D** to detach from session
