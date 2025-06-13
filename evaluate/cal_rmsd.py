from pathlib import Path
import pandas as pd
from loguru import logger
from Bio import PDB
from Bio.pairwise2 import align
from typing import Optional, List
import numpy as np
from peppcbench.multiprocess import OrderedTaskProcessor
from peppcbench.utils import load_job_info

DIGITS = 4


def extract_sequence(chain):
    """Extract the amino acid sequence of the PDB chain and return the residue list"""
    aa_dict = {}
    for res in chain.get_residues():
        res_id = res.get_id()  # Get the full Residue ID
        res_name = res.get_resname()
        if res_name in PDB.Polypeptide.standard_aa_names:
            try:
                aa_dict[res_id] = PDB.Polypeptide.index_to_one(
                    PDB.Polypeptide.three_to_index(res_name)
                )
            except KeyError:
                continue  # Ignore the non-standard amino acid
    return "".join(aa_dict.values()), list(aa_dict.keys())  # Return the full Residue ID


def align_sequences(seq1, seq2):
    """Use Biopython to align sequences"""
    alignment = align.globalxx(seq1, seq2)[0]
    return alignment[0], alignment[1]


def extract_sequence(chain):
    """Extract the amino acid sequence of the PDB chain and return the residue list"""
    aa_dict = {}
    for res in chain.get_residues():
        res_id = res.get_id()  # Get the full Residue ID
        res_name = res.get_resname()
        if res_name in PDB.Polypeptide.standard_aa_names:
            try:
                aa_dict[res_id] = PDB.Polypeptide.index_to_one(
                    PDB.Polypeptide.three_to_index(res_name)
                )
            except KeyError:
                continue  # Ignore the non-standard amino acid
    return "".join(aa_dict.values()), list(aa_dict.keys())  # Return the full Residue ID


def find_continuous_matches(aligned_seq1, aligned_seq2, min_continuous=5):
    """Find all continuous fragments that satisfy the minimum continuous matching length"""
    matches = []
    current_start, current_len = None, 0

    for i, (res1, res2) in enumerate(zip(aligned_seq1, aligned_seq2)):
        if res1 != "-" and res2 != "-" and res1 == res2:
            if current_start is None:
                current_start = i
            current_len += 1
        else:
            if current_len >= min_continuous:
                matches.append((current_start, i - 1))
            current_start, current_len = None, 0

    # Check the last fragment
    if current_len >= min_continuous:
        matches.append((current_start, len(aligned_seq1) - 1))

    return matches


def find_longest_continuous_match(aligned_seq1, aligned_seq2):
    """Find the longest continuous matching fragment in multiple sequence alignment"""
    max_len = 0
    current_len = 0
    max_end = 0

    for i, (res1, res2) in enumerate(zip(aligned_seq1, aligned_seq2)):
        if res1 != "-" and res2 != "-" and res1 == res2:
            current_len += 1
            if current_len > max_len:
                max_len = current_len
                max_end = i
        else:
            current_len = 0

    max_start = max_end - max_len + 1
    if max_len < 5:
        raise ValueError(
            "The continuous fragment is too short, cannot perform reliable RMSD calculation."
        )
    return max_start, max_end


def process_chain_pair(chain_ref, chain_model, mode="all", min_continuous=5):
    """Process PDB chains, keeping continuous matching fragments based on the mode"""
    seq_ref, res_ids_ref = extract_sequence(chain_ref)
    seq_model, res_ids_model = extract_sequence(chain_model)
    aligned_seq_ref, aligned_seq_model = align_sequences(seq_ref, seq_model)

    residue_map = {}

    if mode == "longest":
        start, end = find_longest_continuous_match(aligned_seq_ref, aligned_seq_model)
        matches = [(start, end)]
    elif mode == "all":
        matches = find_continuous_matches(
            aligned_seq_ref, aligned_seq_model, min_continuous
        )
    else:
        raise ValueError("Invalid mode. Use 'longest' or 'all'.")

    residue_map = {}
    for start, end in matches:
        idx_ref, idx_model = 0, 0
        for res_ref, res_model in zip(
            aligned_seq_ref[:start], aligned_seq_model[:start]
        ):
            idx_ref += res_ref != "-"
            idx_model += res_model != "-"

        for res_ref, res_model in zip(
            aligned_seq_ref[start : end + 1], aligned_seq_model[start : end + 1]
        ):
            if res_ref != "-" and res_model != "-":
                residue_map[res_ids_model[idx_model]] = res_ids_ref[idx_ref]
            idx_ref += res_ref != "-"
            idx_model += res_model != "-"

    aligned_seq_ref_cont = aligned_seq_ref[start : end + 1]
    aligned_seq_model_cont = aligned_seq_model[start : end + 1]

    return aligned_seq_ref_cont, aligned_seq_model_cont, residue_map


def filter_atoms(
    chain_ref, chain_model, residue_map, atom_types: Optional[List[str]] = None
):
    ref_atoms, model_atoms = [], []
    for model_res_id, ref_res_id in residue_map.items():
        if not chain_ref.has_id(ref_res_id) or not chain_model.has_id(model_res_id):
            continue  # Ignore the residue if it does not exist in the reference structure

        res_ref = chain_ref[ref_res_id]
        res_model = chain_model[model_res_id]

        for atom_model in res_model:
            if atom_types is None or atom_model.name in atom_types:
                atom_ref = (
                    res_ref[atom_model.name] if atom_model.name in res_ref else None
                )
                if atom_ref is not None:
                    ref_atoms.append(atom_ref)
                    model_atoms.append(atom_model)
    return ref_atoms, model_atoms


def _cal_rmsd(coords1, coords2):
    diff = coords1 - coords2
    return np.sqrt(np.sum(np.sum(diff * diff, axis=1) / coords1.shape[0]))


def cal_rmsd(
    pdb_model_path: Path, pdb_ref_path: Path, atom_types: Optional[List[str]] = None
):
    parser = PDB.PDBParser(QUIET=True)
    struct_ref = parser.get_structure("ref", pdb_ref_path)[0]
    struct_model = parser.get_structure("model", pdb_model_path)[0]

    peptide_chain_ref = struct_ref["A"]  # A 是多肽，B 是蛋白质
    peptide_chain_model = struct_model["A"]
    protein_chain_ref = struct_ref["B"]
    protein_chain_model = struct_model["B"]

    aligned_seq_peptide_ref, aligned_seq_peptide_model, residue_map_peptide = (
        process_chain_pair(peptide_chain_ref, peptide_chain_model)
    )
    aligned_seq_protein_ref, aligned_seq_protein_model, residue_map_protein = (
        process_chain_pair(protein_chain_ref, protein_chain_model)
    )

    ref_atoms_peptide, model_atoms_peptide = filter_atoms(
        peptide_chain_ref,
        peptide_chain_model,
        residue_map_peptide,
        atom_types=atom_types,
    )
    ref_atoms_protein, model_atoms_protein = filter_atoms(
        protein_chain_ref,
        protein_chain_model,
        residue_map_protein,
        atom_types=atom_types,
    )

    # 计算 PEPTIDE RMSD
    superimposer_peptide = PDB.Superimposer()
    superimposer_peptide.set_atoms(ref_atoms_peptide, model_atoms_peptide)

    # 计算蛋白质 RMSD
    superimposer_protein = PDB.Superimposer()
    superimposer_protein.set_atoms(ref_atoms_protein, model_atoms_protein)

    superimposer_protein.apply(model_atoms_peptide)

    # 计算对齐后的多肽 RMSD
    peptide_matrix = np.array([atom.get_coord() for atom in ref_atoms_peptide])
    peptide_aligned_matrix = np.array(
        [atom.get_coord() for atom in model_atoms_peptide]
    )
    peptide_aligned_rmsd = _cal_rmsd(peptide_matrix, peptide_aligned_matrix)

    return {
        "peptide_aligned_rmsd": round(peptide_aligned_rmsd, DIGITS),
        "peptide_rmsd": round(superimposer_peptide.rms, DIGITS),
        "protein_rmsd": round(superimposer_protein.rms, DIGITS),
    }


def process_task(task_args):
    try:
        sample_dir = task_args.pop("sample_dir")
        pdb_model_path = sample_dir / "complex_processed.pdb"
        pdb_ref_path = sample_dir.parents[2] / "complex_processed.pdb"

        scores = cal_rmsd(
            pdb_model_path,
            pdb_ref_path,
            atom_types=["CA", "C", "N", "O"],
        )
        task_args.update(scores)
        return task_args
    except Exception as e:
        logger.error(f"Error processing task {task_args}: {e}")
        task_args.update(
            {
                "peptide_aligned_rmsd": None,
                "peptide_rmsd": None,
                "protein_rmsd": None,
            }
        )
        return task_args


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


if __name__ == "__main__":
    data_dir = Path("./pepdb")
    result_dir = Path("./results")
    model_name = "af3"

    args_list = build_task_args_list(data_dir, model_name)

    task_processor = OrderedTaskProcessor(process_task, num_workers=8)
    results = task_processor.execute_tasks(args_list)
    df = pd.DataFrame(results)
    df.to_csv(result_dir / f"{model_name}_rmsd.csv", index=False)
