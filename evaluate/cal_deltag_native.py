import json
from pathlib import Path
import time
import pyrosetta

from peppcbench.utils import load_job_info, get_ids_from_name
from peppcbench.multiprocess import OrderedTaskProcessor
import pandas as pd
from loguru import logger

DIGITS = 4


def build_task_args_list(data_dir: Path, model_name: str):
    args_list = []
    for job_dir in data_dir.iterdir():
        if not job_dir.is_dir():
            logger.warning(f"{job_dir} is not a directory, skipping")
            continue
        args_list.append(
            {
                "job_id": job_dir.name,
                "model_name": model_name,
                "sample_dir": job_dir,
            }
        )
    return args_list


def cal_deltaG(model_path: Path):
    try:
        assert model_path.exists(), f"PDB file does not exist: {model_path}"
        pose = pyrosetta.pose_from_file(str(model_path))
        time_start = time.time()
        scorefxn = pyrosetta.get_fa_scorefxn()
        total_score = scorefxn(pose)

        interface = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover("A_B")
        interface.set_pack_separated(True)
        interface.set_pack_rounds(3)
        interface.set_compute_interface_sc(True)
        interface.set_compute_packstat(True)
        interface.apply(pose)
        time_end = time.time()
        logger.info(f"Time cost: {time_end - time_start} seconds")
        scores = {
            "dG_separated": round(pose.scores.get("dG_separated", 0), DIGITS),
            "dG_cross": round(pose.scores.get("dG_cross", 0), DIGITS),
            "sc_value": round(pose.scores.get("sc_value", 0), DIGITS),
            "nres_all": int(pose.scores.get("nres_all", 0)),
            "nres_int": int(pose.scores.get("nres_int", 0)),
            "total_score": round(total_score, DIGITS),
        }
        scores["binding_energy"] = scores["dG_cross"] - scores["dG_separated"]
        
        return scores
    except Exception as e:
        logger.error(f"Error processing {model_path}: {e}")
        scores = {
            "dG_separated": None,
            "dG_cross": None,
            "sc_value": None,
            "nres_all": None,
            "nres_int": None,
            "total_score": None,
            "binding_energy": None,
        }
        return scores


def process_task(task_args):
    sample_dir = task_args.pop("sample_dir")
    deltag_json_path = sample_dir / "deltag_native.json"

    force_write = False
    if deltag_json_path.exists() and not force_write:
        logger.info(f"Loading deltag from {deltag_json_path}")
        with open(deltag_json_path, "r") as f:
            scores = json.load(f)
    else:
        scores = cal_deltaG(sample_dir / f"complex_processed.pdb")
        with open(deltag_json_path, "w") as f:
            json.dump(scores, f)

    task_args.update(scores)
    return task_args


if __name__ == "__main__":
    init_options = [
        "-mute",
        "all",
        "-use_input_sc",
        "-ignore_unrecognized_res",
        "-ignore_zero_occupancy",
        "false",
        "-load_PDB_components",
        "false",
        "-no_fconfig",
        "-use_terminal_residues",
        "true",
        "-in:file:silent_struct_type",
        "binary",
    ]  # "-relax:default_repeats", "2",
    logger.info("Initializing PyRosetta with options:", " ".join(init_options))
    pyrosetta.init(" ".join(init_options))

    data_dir = Path("./pepdb")
    result_dir = Path("./results")
    result_dir.mkdir(exist_ok=True)

    model_name = "af3"
    args_list = build_task_args_list(data_dir, model_name)

    task_processor = OrderedTaskProcessor(process_task, num_workers=16)
    results = task_processor.execute_tasks(args_list)
    df = pd.DataFrame(results)
    df.to_csv(result_dir / f"deltaG_native.csv", index=False)
