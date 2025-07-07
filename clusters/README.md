# De redeunt the PepPCSet from scracth

```bash
mamba activate peppcbench

# Step 0: Query PDB for training entries
# - Must be released before 2021-09-30
# - Must contain the term "peptide" in abstract/title
python 0.search_pdb_entry.py \
  --query_file query_train.json \
  --output query_results_train.csv

# Step 1: Fetch PDB mmCIF structures
# - Requires a local copy of the full PDB mmCIF repository
# - Will download or access the required entries listed in the CSV
python 1.build_pdb_data.py

# Step 2: Build the test set (PepPCSet-Full)
python 2.build_pdb_test_data.py

mkdir -p pdb_train_test_data && cd pdb_train_test_data
ln -s ../pdb_data
ln -s ../pdb_test_data
cd ..

bash 3.foldseek_cluster.sh

python 4.build_fasta.py --input_path ./query_results_train.csv --output_path train_chains.fasta
python 4.build_fasta.py --input_path ../job_list.csv --output_path test_chains.fasta

cat train_chains.fasta test_chains.fasta > chains.fasta

bash 5.mmseq_cluster.sh
```
