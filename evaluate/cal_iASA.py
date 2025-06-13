import json
from pathlib import Path
import time
from Bio.PDB import PDBParser

import freesasa
from peppcbench.utils import load_job_info, get_ids_from_name
from peppcbench.multiprocess import OrderedTaskProcessor
import pandas as pd
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")

DIGITS = 4


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


def cal_iASA(peptide_path: Path, receptor_path: Path, complex_path: Path):
    try:
        assert peptide_path.exists(), f"PDB file does not exist: {peptide_path}"
        assert receptor_path.exists(), f"PDB file does not exist: {receptor_path}"
        assert complex_path.exists(), f"PDB file does not exist: {complex_path}"
        time_start = time.time()
        parser = PDBParser(QUIET=True)
        peptide_struct = parser.get_structure("origin", peptide_path)
        receptor_struct = parser.get_structure("origin", receptor_path)
        complex_struct = parser.get_structure("origin", complex_path)

        _sasa_peptide, sasa_peptide_classes = freesasa.calcBioPDB(peptide_struct)
        _sasa_protein, sasa_protein_classes = freesasa.calcBioPDB(receptor_struct)
        _sasa_complex, sasa_complex_classes = freesasa.calcBioPDB(complex_struct)

        sasa_peptide = _sasa_peptide.totalArea()
        sasa_protein = _sasa_protein.totalArea()
        sasa_complex = _sasa_complex.totalArea()

        interface_area = (sasa_peptide + sasa_protein - sasa_complex) / 2
        logger.debug(f"iSASA (S4): {interface_area:.2f} A2")

        interface_area_percentage = (interface_area / sasa_peptide) * 100
        logger.debug(f" iASA%: {interface_area_percentage:.2f} %")

        time_end = time.time()
        logger.info(f"Time cost: {time_end - time_start} seconds")
        return {
            "iASA": interface_area,
            "iASA%": interface_area_percentage,
            "sasa_peptide": sasa_peptide,
            "sasa_protein": sasa_protein,
            "sasa_complex": sasa_complex,
        }
    except Exception as e:
        logger.error(f"Error processing {complex_path}: {e}")
        scores = {
            "iASA": None,
            "iASA%": None,
            "sasa_peptide": None,
            "sasa_protein": None,
            "sasa_complex": None,
        }

        return scores


def process_task(task_args):
    sample_dir = task_args.pop("sample_dir")
    seed_id, sample_id = get_ids_from_name(sample_dir.name)
    task_args.update({"seed_id": seed_id, "sample_id": sample_id})
    iasa_json_path = sample_dir / "iASA.json"
    force_write = False
    if iasa_json_path.exists() and not force_write:
        logger.info(f"Lwoading iASA from {iasa_json_path}")
        with open(iasa_json_path, "r") as f:
            scores = json.load(f)
    else:
        peptide_path = sample_dir / "peptide.pdb"
        receptor_path = sample_dir / "protein.pdb"
        complex_path = sample_dir / "complex.pdb"

        scores = cal_iASA(peptide_path, receptor_path, complex_path)
        with open(iasa_json_path, "w") as f:
            json.dump(scores, f)

    task_args.update(scores)
    return task_args


if __name__ == "__main__":
    data_dir = Path("./pepdb")
    result_dir = Path("./results")

    model_name = "af3"
    args_list = build_task_args_list(data_dir, model_name)

    task_processor = OrderedTaskProcessor(process_task, num_workers=24)
    results = task_processor.execute_tasks(args_list)
    df = pd.DataFrame(results)
    df.to_csv(result_dir / f"{model_name}_iASA.csv", index=False)
