from pathlib import Path
import subprocess
import pandas as pd
from loguru import logger

from peppcbench.utils import load_job_info, get_ids_from_name, load_json
from peppcbench.multiprocess import OrderedTaskProcessor

import sys

logger.remove()
logger.add(sys.stdout, level="INFO")

DOCKQ_SCRIPT_PATH = "./scripts/run_dockq.sh"


def get_dockQ_score(json_dict: dict):
    try:
        best_result = json_dict["best_result"]
        assert "AB" in best_result, "DockQ json file does not contain AB"
        scores = {}
        scores["DockQ"] = best_result["AB"]["DockQ"]
        scores["F1"] = best_result["AB"]["F1"]
        scores["iRMSD"] = best_result["AB"]["iRMSD"]
        scores["LRMSD"] = best_result["AB"]["LRMSD"]
        scores["fnat"] = best_result["AB"]["fnat"]
        scores["clashes"] = best_result["AB"]["clashes"]
        return scores
    except Exception as e:
        raise ValueError(f"Error in getting dockQ score: {e}")


def get_cmd(pdb_model: Path, pdb_ref: Path, json_file_name: str):
    """Please make sure the pdb with chain A (peptide) and chain B (protein)"""
    return [
        "bash",
        DOCKQ_SCRIPT_PATH,
        str(pdb_model.resolve()),
        str(pdb_ref.resolve()),
        json_file_name,
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

        task_args.update(
            {
                "seed_id": seed_id,
                "sample_id": sample_id,
                "job_id": job_id,
            }
        )

        pdb_model_path = sample_dir / "complex_processed.pdb"
        pdb_ref_path = sample_dir.parents[2] / "complex_processed.pdb"
        assert (
            pdb_model_path.exists() and pdb_ref_path.exists()
        ), f"PDB files do not exist: {pdb_model_path} {pdb_ref_path}"

        dockq_json_path = pdb_model_path.parent / "dockq.json"
        cmd = get_cmd(pdb_model_path, pdb_ref_path, dockq_json_path.name)
        
        forse_write = False
        if dockq_json_path.exists() and not forse_write:
            logger.info(
                f"DockQ (dockq.json) for {str(pdb_model_path)} and {str(pdb_ref_path)} already exists"
            )
            scores = get_dockQ_score(load_json(dockq_json_path))
            task_args.update(scores)
            return task_args
        else:
            subprocess.run(cmd, check=True)
            scores = get_dockQ_score(load_json(dockq_json_path))
            task_args.update(scores)
            return task_args
    except Exception as e:
        raise ValueError(f"Error in process_task: {e}")


if __name__ == "__main__":
    data_dir = Path("./pepdb")
    result_dir = Path("./results")
    model_name = "af3"
    args_list = build_task_args_list(data_dir, model_name)
    task_processor = OrderedTaskProcessor(process_task, num_workers=8)
    results = task_processor.execute_tasks(args_list)
    df = pd.DataFrame(results)
    df.to_csv(result_dir / f"{model_name}_dockq.csv", index=False)
