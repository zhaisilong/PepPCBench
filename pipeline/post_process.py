import json
from pathlib import Path
from loguru import logger
from peppcbench.multiprocess import OrderedTaskProcessor
from peppcbench.utils import load_job_info
from peppcbench.preprocess import prepare_pdb, cif2pdb, process_pdb_pair_chains
import traceback
from time import sleep
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")


def build_task_args_list(data_dir: Path, model_name: str):
    args_list = []
    for job_dir in data_dir.iterdir():
        if not job_dir.is_dir():
            continue

        job_info = load_job_info(job_dir)
        job_name = job_info["job_name"]
        infer_done_flag = job_dir / model_name / "infer.done"
        if not infer_done_flag.exists():
            logger.warning(f"{job_name} was not finished, skipping")
            continue

        for sample_dir in (job_dir / "af3" / job_name.lower()).glob("seed-*_sample-*"):
            if not sample_dir.is_dir():
                continue

            args_list.append(
                {
                    "job_id": job_dir.name,
                    "model_name": model_name,
                    "sample_dir": sample_dir,
                }
            )
    return args_list


def process_task(task_args: dict):
    if task_args["model_name"] == "af3":
        return _process_task_af3(task_args)


def _process_task_af3(task_args: dict) -> bool:
    try:
        sample_dir: Path = task_args["sample_dir"]
        job_dir: Path = sample_dir.parents[2]
        job_info_path: Path = job_dir / "job_info.json"
        job_info: dict = json.loads(job_info_path.read_text())

        peptide_chains = job_info.get("peptide_chains", None)
        protein_chains = job_info.get("protein_chains", None)

        structure_path_cif = sample_dir / "model.cif"
        structure_path_pdb = cif2pdb(structure_path_cif, force_write=False)

        prepare_pdb(
            pdb_path=structure_path_pdb,
            peptide_chains=peptide_chains,
            protein_chains=protein_chains,
            force_write=False,
        )

        process_pdb_pair_chains(
            pdb_path=structure_path_pdb,
            pdb_ref_path=job_dir / f"{task_args['job_id'][:4]}.pdb",
            peptide_chains=peptide_chains,
            protein_chains=protein_chains,
            force_write=False,
        )
        return True

    except Exception as e:
        logger.error(f"Error processing task: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    target_dir = Path("./pepdb")
    model_name = "af3"

    args_list = build_task_args_list(target_dir, model_name)
    task_processor = OrderedTaskProcessor(
        task_function=process_task,
        num_workers=12,
    )
    success = task_processor.execute_tasks(args_list)
    print(f"Success Rate: {sum(success) / len(success) * 100}%")
