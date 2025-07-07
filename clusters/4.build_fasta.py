import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import click

# 全局资源
fasta_dir = Path("fasta_download")
fasta_dir.mkdir(exist_ok=True)
lock = Lock()

@click.command()
@click.option("--input_path", type=str, required=True, help="Path to input CSV with pdb_id column")
@click.option("--output_path", type=str, required=True, help="Path to save combined FASTA output")
def main(input_path, output_path):
    df = pd.read_csv(input_path)
    pdb_ids = df["pdb_id"].dropna().unique().tolist()
    all_chains = []

    def download_and_process_fasta(pdb_id):
        """
        下载并处理单个 PDB ID 的 FASTA 序列，清洗后加入 all_chains
        """
        pdb_id_clean = pdb_id.strip().upper()
        fasta_path = fasta_dir / f"{pdb_id_clean}.fasta"

        # 下载
        if not fasta_path.exists():
            try:
                url = f"https://www.rcsb.org/fasta/entry/{pdb_id_clean}/download"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(fasta_path, "w") as f:
                        f.write(response.text)
                else:
                    print(f"[ERROR] Failed to download {pdb_id_clean}: HTTP {response.status_code}")
                    return
            except Exception as e:
                print(f"[ERROR] Exception downloading {pdb_id_clean}: {e}")
                return

        # 解析
        try:
            records = list(SeqIO.parse(fasta_path, "fasta"))
            with lock:
                for record in records:
                    record.id = record.id.replace("|", " |")
                    clean_seq = str(record.seq).replace(" ", "").replace("\n", "").replace("\r", "")
                    record.seq = Seq(clean_seq)
                    if len(clean_seq) >= 50:
                        all_chains.append(record)
        except Exception as e:
            print(f"[ERROR] Failed to parse {pdb_id_clean}: {e}")

    # 多线程处理
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(download_and_process_fasta, pdb_ids), total=len(pdb_ids)))

    # 写入输出
    SeqIO.write(all_chains, output_path, "fasta")
    print(f"[DONE] Processed {len(all_chains)} chains. Saved to {output_path}")

if __name__ == "__main__":
    main()