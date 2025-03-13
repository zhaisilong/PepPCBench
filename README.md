# PepPCBench

Comprehensive Benchmark for Protein-Peptide Complex Structure Prediction with All-Atom Protein Folding Neural Networks

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

#### 

## Citations

```bibtex
@article{}
```
