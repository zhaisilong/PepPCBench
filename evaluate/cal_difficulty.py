import json
from pathlib import Path
import MDAnalysis as mda
import PeptideBuilder
import Bio
from scipy.spatial import cKDTree
from MDAnalysis.analysis.align import rotation_matrix
from peppcbench.multiprocess import OrderedTaskProcessor
import pandas as pd
from loguru import logger
import tempfile
import os
import sys
import numpy as np
from pymol import cmd

logger.remove()
logger.add(sys.stdout, level="DEBUG")

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


def pymol_rmsd(bound_pdb, model_pdb, selection="name N+CA+C"):
    """
    使用 PyMOL 计算 RMSD，对象名避免使用保留字
    """
    cmd.reinitialize()
    cmd.load(bound_pdb, "bound")
    cmd.load(model_pdb, "ref")  # 替换掉 'model'

    # 注意这里也要用 'ref'
    rmsd = cmd.align(f"bound and {selection}", f"ref and {selection}")[0]
    return rmsd


def classify_difficulty(bound_pdb_path, sequence):
    """
    生成理想构象（helical、extended、ppii），用 PyMOL 计算 RMSD，并判断难度等级。
    返回包含所有 RMSD 和 difficulty 的字典。

    参数:
        bound_pdb_path (str): bound.pdb 路径
        sequence (str): 氨基酸序列（单字母）

    返回:
        dict: {
            'helical': float,
            'extended': float,
            'ppii': float,
            'difficulty': str
        }
    """
    temp_dir = tempfile.mkdtemp()
    io = Bio.PDB.PDBIO()

    try:
        # === 构象生成 ===
        helical_pdb = os.path.join(temp_dir, "helical.pdb")
        extended_pdb = os.path.join(temp_dir, "extended.pdb")
        ppii_pdb = os.path.join(temp_dir, "ppii.pdb")

        helix_struct = PeptideBuilder.make_structure(
            sequence, phi=[-57] * len(sequence), psi_im1=[-47] * len(sequence)
        )
        io.set_structure(helix_struct)
        io.save(helical_pdb)

        ext_struct = PeptideBuilder.make_structure(
            sequence, phi=[-140] * len(sequence), psi_im1=[130] * len(sequence)
        )
        io.set_structure(ext_struct)
        io.save(extended_pdb)

        ppii_struct = PeptideBuilder.make_structure(
            sequence, phi=[-78] * len(sequence), psi_im1=[149] * len(sequence)
        )
        io.set_structure(ppii_struct)
        io.save(ppii_pdb)

        # === PyMOL 计算 RMSD ===
        rmsd_vals = {
            "helical": pymol_rmsd(bound_pdb_path, helical_pdb),
            "extended": pymol_rmsd(bound_pdb_path, extended_pdb),
            "ppii": pymol_rmsd(bound_pdb_path, ppii_pdb),
        }

        # === 分类判断 ===
        h = rmsd_vals["helical"]
        e = rmsd_vals["extended"]
        if (h <= 4) or (e <= 4):
            difficulty = "Easy"
        elif (h > 4 and h <= 8) and (e > 4 and e <= 8):
            difficulty = "Medium"
        else:
            difficulty = "Difficult"

        rmsd_vals["difficulty"] = difficulty

    except Exception as e:
        rmsd_vals = {
            "helical": None,
            "extended": None,
            "ppii": None,
            "difficulty": "Unclassified",
            "error": str(e),
        }

    finally:
        for f in [helical_pdb, extended_pdb, ppii_pdb]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(temp_dir)

    return rmsd_vals


def process_task(task_args):
    sample_dir = task_args.pop("sample_dir")
    difficulty_json_path = sample_dir / "difficulty.json"
    job_info_path = sample_dir / "job_info.json"
    job_info = json.load(open(job_info_path, "r"))
    peptide_chains = job_info["peptide_chains"]
    peptide_sequence = job_info[peptide_chains]["sequence"]
    logger.debug(f"sample_dir: {sample_dir}, sequence: {peptide_sequence}")
    force_write = True
    if difficulty_json_path.exists() and not force_write:
        logger.info(f"Loading difficulty from {difficulty_json_path}")
        with open(difficulty_json_path, "r") as f:
            scores = json.load(f)
    else:
        scores = classify_difficulty(
            bound_pdb_path=sample_dir / f"peptide.pdb",
            sequence=peptide_sequence,
        )
        with open(difficulty_json_path, "w") as f:
            json.dump(scores, f)

    task_args.update(scores)
    return task_args


if __name__ == "__main__":
    data_dir = Path("./pepdb")
    result_dir = Path("./results")
    result_dir.mkdir(exist_ok=True)

    model_name = "af3"
    args_list = build_task_args_list(data_dir, model_name)

    task_processor = OrderedTaskProcessor(process_task, num_workers=8)
    results = task_processor.execute_tasks(args_list)
    df = pd.DataFrame(results)
    df.to_csv(result_dir / f"{model_name}_difficulty.csv", index=False)
