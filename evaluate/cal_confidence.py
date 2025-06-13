from pathlib import Path
import pandas as pd
from copy import deepcopy
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


def _process_task_af3(task_args: dict):
    seed_id, sample_id = get_ids_from_name(task_args["sample_dir"].name)

    # output = get_confidence(task_args["sample_dir"])
    output = get_confidence_extra(task_args["sample_dir"], force_write=False)

    result = deepcopy(task_args)
    result.update(
        {
            "seed_id": seed_id,
            "sample_id": sample_id,
        }
    )
    result.update(output)
    result.pop("sample_dir")
    return result


if __name__ == "__main__":
    data_dir = Path("./pepdb")
    result_dir = Path("./results")
    model_name = "af3"
    args_list = build_task_args_list(data_dir, model_name)[:1]
    task_processor = OrderedTaskProcessor(
        task_function=process_task,
        num_workers=8,
    )
    results = task_processor.execute_tasks(args_list)
    df = pd.DataFrame(results)
    df.to_csv(result_dir / f"{model_name}_confidence_tmp.csv", index=False)
