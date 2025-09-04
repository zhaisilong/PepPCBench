# PepPCBench

PepPCBench: A Comprehensive Benchmark for Protein-Peptide Complex Structure Prediction with AlphaFold3

We welcome feedback from the community!
For questions, issues, or dataset requests, please open an [Issue](https://github.com/zhaisilong/PepPCBench/issues) or contact us directly at `zhaisilong@outlook.com`.

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

- A seperated `tools` directory is used to store all the external tools.

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

#### PyRosetta

- Using PyRosetta to calculate the deltaG of the predicted the binding energy of the peptide-protein complex

```bash
pip install pyrosetta-installer
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

## Experiments

All code has been optimized for multiprocessing and tested on a multi-GPU machine, specifically the A800, using Python version 3.12. For high-throughput processing, please refer to `peppcbench/pipeline.py` and `peppcbench/multiprocess.py` for further details.

### Workflow

```bash
# Build Data
python pipeline/build_data.py

# Extract Job Info
python pipeline/extract_job_info.py

# Generate Config
python pipeline/gen_config.py

# Run Model
python pipeline/run_model.py
## with specific GPU, default is 3
python pipeline/run_model.py --gpu_id=0

# Supervise Workflow
python pipeline/summary_jobs.py  # results are saved in `./results/job_summary.png` and `./results/job_summary.csv`

# Post Process
python pipeline/post_process.py
```

### Metrics

- The output files are in `./results`

```bash
python evaluate/cal_confidence.py  # results are saved in `./results/af3_confidence.csv`
python evaluate/cal_dockq.py  # results are saved in `./results/af3_dockq.csv`
python evaluate/cal_rmsd.py  # results are saved in `./results/af3_rmsd.csv`
python evaluate/cal_deltag.py  # results are saved in `./results/af3_deltag.csv`

# if you want to relax the predicted complex, you can run the following command:
python pipeline/relax.py
# then you can run the following command to calculate the deltaG of the predicted complex after relaxation
python evaluate/cal_deltag.py
```

### Analysis

- some useful notebooks are in `./notebooks` to analyze the results
- Contents
  - `contact_map.ipynb`: visualize the contact map of the predicted complex
  - `ranking_power.ipynb`: visualize the ranking power of different scoring functions for the predicted complex

## Some Important Notes

1. The calculation of `actifpTM` is based on additional outputs provided by AlphaFold3. The corresponding extraction code can be found in the `patches/actifpTM` directory and the notebook `./notebooks/contact_map.ipynb`.

## Citations

```bibtex
@article{zhaiPepPCBenchComprehensiveBenchmarking2025,
  title = {{{PepPCBench}} Is a {{Comprehensive Benchmarking Framework}} for {{Protein}}–{{Peptide Complex Structure Prediction}}},
  author = {Zhai, Silong and Zhao, Huifeng and Wang, Jike and Lin, Shaolong and Liu, Tiantao and Gu, Shukai and Jiang, Dejun and Liu, Huanxiang and Kang, Yu and Yao, Xiaojun and Hou, Tingjun},
  date = {2025-08-25},
  journaltitle = {Journal of Chemical Information and Modeling},
  shortjournal = {J. Chem. Inf. Model.},
  volume = {65},
  number = {16},
  pages = {8497--8513},
  publisher = {American Chemical Society},
  issn = {1549-9596},
  doi = {10.1021/acs.jcim.5c01084},
  url = {https://doi.org/10.1021/acs.jcim.5c01084}
}
```
