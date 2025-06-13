import json
from pathlib import Path
import subprocess
import traceback
from loguru import logger

from peppcbench.utils import load_job_info, get_ids_from_name, load_json
from peppcbench.multiprocess import OrderedTaskProcessor
from peppcbench.preprocess import process_pdb_pair_chains_for_relax
from time import sleep
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")

RELAX_SCRIPT_PATH = "./scripts/run_relax.sh"
CUDA_ID = 2


def get_cmd(input_path: Path, output_path: Path):
    """Please make sure the pdb with chain A (peptide) and chain B (protein)"""
    return [
        "bash",
        RELAX_SCRIPT_PATH,
        str(input_path.resolve()),
        str(output_path.resolve()),
        str(CUDA_ID),
    ]


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
    try:
        sample_dir: Path = task_args.pop("sample_dir")
        seed_id, sample_id = get_ids_from_name(sample_dir.name)
        job_id = task_args["job_id"]
        job_dir: Path = sample_dir.parents[2]
        job_info_path: Path = job_dir / "job_info.json"
        job_info: dict = json.loads(job_info_path.read_text())
        peptide_chains = job_info.get("peptide_chains", None)
        protein_chains = job_info.get("protein_chains", None)

        task_args.update(
            {
                "seed_id": seed_id,
                "sample_id": sample_id,
                "job_id": job_id,
            }
        )

        pdb_model_path = sample_dir / "complex.pdb"

        assert pdb_model_path.exists(), f"PDB files do not exist: {pdb_model_path}"
        relaxed_pdb_path = pdb_model_path.parent / "complex_relaxed.pdb"

        cmd = get_cmd(pdb_model_path, relaxed_pdb_path)

        force_write = False
        if relaxed_pdb_path.exists() and not force_write:
            logger.info(f"Relaxed PDB for {str(pdb_model_path)} already exists")
        else:
            logger.debug(f"Relaxing CMD is {cmd}")
            subprocess.run(cmd, check=True)

        process_pdb_pair_chains_for_relax(
            pdb_path=relaxed_pdb_path,
            pdb_ref_path=job_dir / f"{task_args['job_id'][:4]}.pdb",
            peptide_chains=peptide_chains,
            protein_chains=protein_chains,
            force_write=force_write,
            suffix_name="complex_relaxed_processed",
        )
        return True

    except Exception as e:
        logger.error(f"Error in process {task_args}: {e}")
        logger.error(traceback.format_exc())
        sleep(1000)
        return False


if __name__ == "__main__":
    data_dir = Path("./pepdb")
    model_name = "af3"
    args_list = build_task_args_list(data_dir, model_name)
    task_processor = OrderedTaskProcessor(process_task, num_workers=1, timeout=300)
    success = task_processor.execute_tasks(args_list)
    logger.info(f"Success Rate: {sum(success) / len(success) * 100}%")
