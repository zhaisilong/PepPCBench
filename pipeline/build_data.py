from pathlib import Path
from loguru import logger
import pandas as pd
from peppcbench.multiprocess import OrderedTaskProcessor
from peppcbench.preprocess import prepare_pdb, download_pdb, cif2pdb
import traceback

import sys

logger.remove()
logger.add(sys.stdout, level="INFO")


def build_task_args_list(df: pd.DataFrame, target_dir: Path):
    arg_list = []
    for _, row in df.iterrows():
        job_name = row.get("job_name")
        pdb_id = row.get("pdb_id")
        job_dir = target_dir / job_name
        peptide_chains = row.get("peptide_chains")
        protein_chains = row.get("protein_chains")

        arg_list.append(
            {
                "job_name": job_name,
                "pdb_id": pdb_id,
                "job_dir": job_dir,
                "peptide_chains": peptide_chains,
                "protein_chains": protein_chains,
            }
        )
    return arg_list


def process_task(task_args: dict):
    try:
        job_name = task_args.get("job_name")
        pdb_id = task_args.get("pdb_id")
        job_dir = task_args.get("job_dir")
        peptide_chains = task_args.get("peptide_chains")
        protein_chains = task_args.get("protein_chains")
        job_dir.mkdir(parents=True, exist_ok=True)

        download_pdb(pdb_id=pdb_id, data_dir=job_dir, force_write=False)

        structure_path_cif = job_dir / f"{job_name[:4]}.cif"
        structure_path_pdb = cif2pdb(structure_path_cif, force_write=False)

        prepare_pdb(
            pdb_path=structure_path_pdb,
            peptide_chains=peptide_chains,
            protein_chains=protein_chains,
            force_write=False,
        )

    except Exception as e:
        logger.error(f"Error processing task: {e}")
        traceback.print_exc()
        return False
    return True


if __name__ == "__main__":

    job_list_path = Path("./job_list.csv")
    target_dir = Path("./pepdb")
    num_workers = 8

    df = pd.read_csv(job_list_path)
    args_list = build_task_args_list(df, target_dir)
    task_processor = OrderedTaskProcessor(
        task_function=process_task,
        num_workers=num_workers,
    )
    task_processor.execute_tasks(args_list)
