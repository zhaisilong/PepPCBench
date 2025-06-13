from pathlib import Path
from loguru import logger
from peppcbench.multiprocess import OrderedTaskProcessor
from peppcbench.utils import load_job_info
from shutil import copyfile
import traceback
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")


def build_task_args_list(source_dir: Path, target_dir: Path, model_name: str):
    args_list = []
    for job_dir in source_dir.iterdir():
        if not job_dir.is_dir():
            continue

        job_info = load_job_info(job_dir)
        job_name = job_info["job_name"]
        infer_done_flag = job_dir / model_name / "infer.done"
        if not infer_done_flag.exists():
            logger.warning(f"{job_name} was not finished, skipping")
            continue

        for sample_dir in (job_dir / model_name / job_name.lower()).glob(
            "seed-*_sample-*"
        ):
            if not sample_dir.is_dir():
                continue

            args_list.append(
                {
                    "job_id": job_dir.name,
                    "model_name": model_name,
                    "sample_dir": sample_dir,
                    "target_sample_dir": target_dir
                    / job_dir.name
                    / model_name
                    / job_name.lower()
                    / sample_dir.name,
                }
            )
    return args_list


def process_task(task_args: dict):
    if task_args["model_name"] == "af3":
        return _process_task_af3(task_args)


def _process_task_af3(task_args: dict) -> bool:
    try:
        sample_dir: Path = task_args["sample_dir"]
        target_sample_dir: Path = task_args["target_sample_dir"]
        target_sample_dir.mkdir(parents=True, exist_ok=True)
        copyfile(
            sample_dir / "complex_relaxed.pdb",
            target_sample_dir / "complex_relaxed.pdb",
        )
        copyfile(
            sample_dir / "complex_relaxed_processed.pdb",
            target_sample_dir / "complex_relaxed_processed.pdb",
        )
        copyfile(
            sample_dir / "chain_mapping_relaxed.json",
            target_sample_dir / "chain_mapping_relaxed.json",
        )
        return True

    except Exception as e:
        logger.error(f"Error processing task: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    source_dir = Path("./pepdb")
    target_dir = Path("./fix")
    model_name = "af3"

    args_list = build_task_args_list(source_dir, target_dir, model_name)
    task_processor = OrderedTaskProcessor(
        task_function=process_task,
        num_workers=24,
    )
    success = task_processor.execute_tasks(args_list)
    print(f"Success Rate: {sum(success) / len(success) * 100}%")
