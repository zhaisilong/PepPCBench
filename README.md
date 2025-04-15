# PepPCBench

PepPCBench is a Comprehensive Benchmark for Protein-Peptide Complex Structure Prediction with AlphaFold3.

⚠️ Under Peer Review | Data & Scripts Coming Soon

We welcome community feedback! For questions or dataset requests, open an ​Issue​ or contact zhaisilong@outlook.com.

## Setups

```bash
mamba create -n peppcbench python=3.12
mamba activate peppcbench
mamba install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia

mamba install -c schrodinger pymol openbabel
mamba install -c conda-forge -c nvidia pdbfixer

pip install biopython loguru tqdm pandas "numpy<2" ipykernel matplotlib seaborn fire pdb-tools gemmi gpustat absl-py
pip install -e .
```

### Optional

#### Maxit

- Helix3 Use it
- <https://sw-tools.rcsb.org/apps/MAXIT/source.html>

```bash
tar xvf maxit-v11.300-prod-src.tar
cd maxit-v11.300-prod-src
make binary

export RCSBROOT=$HOME/tools/maxit/maxit-v11.300-prod-src
export PATH=$RCSBROOT/bin:$PATH

# or
alias set_maxit="export RCSBROOT=${HOME}/tools/maxit/maxit-v11.300-prod-src && export PATH=${RCSBROOT}/bin:${PATH}"
```

#### DockQ

```bash
git clone https://github.com/bjornwallner/DockQ.git

cd DockQ
python -m pip install -e .
bash run_test.sh
```

## Experiments

All code has been optimized for multiprocessing and tested on a multi-GPU machine, specifically the A800, using Python version 3.12. For high-throughput processing, please refer to `peppcbench/pipeline.py` and `peppcbench/multiprocess.py` for further details.

### Workflow

```bash
# Build Data
python -m pipeline.build_data

# Extract Job Info
python -m pipeline.extract_job_info

# Generate Config
python -m pipeline.gen_config

# Run Model
python -m pipeline.run_model

# Supervise Workflow
python -m pipeline.summary_jobs

# Evaluate
python -m pipeline.evaluate
```

### Analysis

- some useful notebooks are in `./notebooks` to analyze the results

## Citations

```bibtex
@article {Zhai2025.04.08.647699,
	author = {Zhai, Silong and Zhao, Huifeng and Wang, Jike and Lin, Shaolong and Liu, Tiantao and Jiang, Dejun and Liu, Huanxiang and Kang, Yu and Yao, Xiaojun and Hou, Tingjun},
	title = {PepPCBench is a Comprehensive Benchmark for Protein-Peptide Complex Structure Prediction with AlphaFold3},
	elocation-id = {2025.04.08.647699},
	year = {2025},
	doi = {10.1101/2025.04.08.647699},
	publisher = {Cold Spring Harbor Laboratory},
	eprint = {https://www.biorxiv.org/content/early/2025/04/13/2025.04.08.647699.full.pdf},
	journal = {bioRxiv}
}
```
