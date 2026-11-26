# E1 CO2 Fixing



- [Chemistry](./docs/chemistry.md)

## Run

This project uses [uv](https://docs.astral.sh/uv/) (see [pyproject.toml](./pyproject.toml)).
Some helper scripts are in [scripts/](./scripts/).
Use [scripts/install.sh](./scripts/install.sh) to set everything up.

### GPU Machine

Installation on Ubuntu server 22.04 with Quadro RTX4000 (8 GB GPU memory), 32 GiB RAM, and 8 vCPU:

1. Update ubuntu `sudo apt update && sudo apt upgrade -y`
2. Install drivers `sudo ubuntu-drivers install --gpgpu`
3. Install build dependencies `sudo apt install build-essential python3-dev -y`
4. Reboot
5. Check drivers branch `dpkg -l | grep -E 'nvidia-driver|nvidia-utils|linux-modules-nvidia'`
6. Install the correct utils, e.g. `sudo apt install nvidia-utils-580-server`
7. Clone repo `git clone https://github.com/mRcSchwering/magic-soup.git`
8. Install uv `curl -LsSf https://astral.sh/uv/install.sh | sh`
9. Setup venv `cd magic-soup/ && uv venv --python 3.12`
10. Reboot
11. Install magic-soup `uv sync --group dev --no-sources-package torch --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match`
12. Check driver and supported CUDA version `nvidia-smi`
13. Check CUDA works with pytorch `run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"`