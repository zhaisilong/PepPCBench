from pathlib import Path
import pandas as pd
from loguru import logger
from peppcbench.utils import load_job_info, get_ids_from_name
from peppcbench.multiprocess import OrderedTaskProcessor
from pipeline.confidence import get_confidence, get_confidence_extra
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
            pocket_file = sample_dir / "pocket_relaxed_fix.pdb"
            ligand_file = sample_dir / "peptide_relaxed_fix.pdb"
            seed_id, sample_id = get_ids_from_name(sample_dir.name)
            args_list.append(
                {
                    "pdb_id": f"{job_dir.name}_{seed_id}_{sample_id}",
                    "pocket_file": str(pocket_file.resolve()),
                    "ligand_file": str(ligand_file.resolve()),
                }
            )
    return args_list


if __name__ == "__main__":
    data_dir = Path("./pepdb")
    result_dir = Path("./results")
    model_name = "af3"
    args_list = build_task_args_list(data_dir, model_name)
    df = pd.DataFrame(args_list)
    df.to_csv(result_dir / f"{model_name}_deepppiscore_inputs_relax.csv", index=False)
