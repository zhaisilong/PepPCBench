import os
from pathlib import Path
from io import StringIO
import re
import subprocess
import gemmi
from loguru import logger
from typing import Any
from pymol import cmd
from pdbfixer import PDBFixer
from Bio.PDB.MMCIFParser import MMCIFParser
from pdbfixer.pdbfixer import Template
from openmm.app import PDBFile
from pdbtools import (
    pdb_wc,
    pdb_delelem,
    pdb_fromcif,
)
import importlib.util
from Bio.PDB import PDBList
import openmm.app as app

PDB_LIST = PDBList(server="https://files.wwpdb.org")


def download_pdb(pdb_id: str, data_dir: Path, force_write: bool = False):
    try:
        logger.info(f"Downloading PDB ID {pdb_id} to {data_dir}")
        PDB_LIST.retrieve_pdb_file(
            pdb_id, pdir=str(data_dir), file_format="mmCif", overwrite=force_write
        )
        logger.info(f"Successfully downloaded {pdb_id}")
    except Exception as e:
        raise ValueError(f"Error downloading PDB ID {pdb_id}: {e}")


def write_structure_with_check(
    structure: str, save_path: Path, force_write: bool = False
):
    if save_path.exists() and not force_write:
        logger.info(f"File {save_path} already exists")
        return
    cmd.save(str(save_path), structure)


def split_structure(
    structure_path: Path,
    peptide_chains: str,
    protein_chains: str,
    radius: float = 10.0,
    get_complex: bool = True,
    get_protein: bool = True,
    get_pocket: bool = True,
    get_peptide: bool = True,
    get_peptide_pocket: bool = True,
    force_split: bool = False,
):
    structure_path = structure_path.resolve()
    peptide_save_path = structure_path.parent / "peptide.pdb"
    pocket_save_path = structure_path.parent / "pocket.pdb"
    complex_save_path = structure_path.parent / "complex.pdb"
    protein_save_path = structure_path.parent / "protein.pdb"
    peptide_pocket_save_path = structure_path.parent / "peptide_pocket.pdb"

    cmd.reinitialize()
    cmd.load(structure_path)
    peptide_chains_str = " or ".join(
        f"chain {chain}" for chain in peptide_chains.split(":")
    )
    protein_chains_str = " or ".join(
        f"chain {chain}" for chain in protein_chains.split(":")
    )

    cmd.select("pep", peptide_chains_str)
    cmd.select("prot", protein_chains_str)

    cmd.select("pocket", f"(byres (br. pep around {radius})) and prot")
    cmd.select("complex", "pep or prot")
    cmd.select("peptide_pocket", "pep or pocket")

    if get_complex:
        write_structure_with_check("complex", complex_save_path, force_split)
    if get_protein:
        write_structure_with_check("prot", protein_save_path, force_split)
    if get_pocket:
        write_structure_with_check("pocket", pocket_save_path, force_split)
    if get_peptide:
        write_structure_with_check("pep", peptide_save_path, force_split)
    if get_peptide_pocket:
        write_structure_with_check(
            "peptide_pocket", peptide_pocket_save_path, force_split
        )


def cif2pdb(cif_path: Path, force_write: bool = False):
    assert cif_path.suffix == ".cif", "the cif file should have a .cif suffix"
    try:
        pdb_path = cif_path.with_suffix(".pdb")
        if pdb_path.exists() and not force_write:
            logger.info(f"{pdb_path} exists skip")
            return pdb_path
        maxit_cmd = [
            "maxit",
            "-input",
            str(cif_path),
            "-output",
            str(pdb_path),
            "-o",
            "2",
        ]
        logger.debug(f"Maxit CIF to PDB: {' '.join(maxit_cmd)}")
        result = subprocess.run(maxit_cmd, capture_output=True, text=True, check=True)
        logger.debug(result.stdout)
        return pdb_path
    except Exception as e:
        raise RuntimeError(f"Failed to convert CIF to PDB: {cif_path} since {e}")


def pdb2cif(pdb_path: Path, force_write: bool = False) -> Path:
    assert pdb_path.suffix == ".pdb", "the pdb file should have a .pdb suffix"
    try:
        cif_path = pdb_path.with_suffix(".cif")
        if cif_path.exists() and not force_write:
            logger.info(f"{cif_path} exists skip")
            return cif_path
        maxit_cmd = [
            "maxit",
            "-input",
            str(pdb_path),
            "-output",
            str(cif_path),
            "-o",
            "1",
        ]
        logger.debug(f"Maxit PDB to CIF: {' '.join(maxit_cmd)}")
        result = subprocess.run(maxit_cmd, capture_output=True, text=True, check=True)
        logger.debug(result.stdout)
        return cif_path
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDB to CIF: {pdb_path} since {e}")


def run_pdbtools(pdb_io: StringIO, func_name: Any, *args, **kwargs) -> str:
    return "".join(func_name.run(pdb_io, *args, **kwargs))


class _PDBFixer(PDBFixer):
    def __init__(self, file_io):
        self._initializeFromPDB(file_io)
        # Check the structure has some atoms in it.
        atoms = list(self.topology.atoms())
        if len(atoms) == 0:
            raise Exception("Structure contains no atoms.")

        # Keep a cache of downloaded CCD definitions
        self._ccdCache = {}

        # Load the templates.

        self.templates = {}
        templatesPath = os.path.join(
            os.path.dirname(importlib.util.find_spec("pdbfixer").origin), "templates"
        )
        for file in os.listdir(templatesPath):
            templatePdb = app.PDBFile(os.path.join(templatesPath, file))
            name = next(templatePdb.topology.residues()).name
            self.templates[name] = Template(templatePdb.topology, templatePdb.positions)


def fix_pdb(pdb_io: StringIO, save_path: Path):
    fixer = _PDBFixer(pdb_io)

    # Find missing residues
    missing_residues = fixer.findMissingResidues()
    if missing_residues:
        logger.warning(f"Found missing residues: {missing_residues}")

    # Find nonstandard residues
    nonstandard_residues = fixer.findNonstandardResidues()
    if nonstandard_residues:
        logger.warning(f"Found nonstandard residues: {nonstandard_residues}")
        fixer.replaceNonstandardResidues()

    # Find missing atoms
    missing_atoms = fixer.findMissingAtoms()
    if missing_atoms:
        logger.warning(f"Found missing atoms: {missing_atoms}")
        fixer.addMissingAtoms()

    fixer.removeHeterogens(False)

    with save_path.open("w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)


def prepare_pdb(
    pdb_path: Path,
    peptide_chains: str,
    protein_chains: str,
    force_write: bool = False,
    verbose: bool = False,
):
    if verbose:
        with pdb_path.open("r") as f:
            has_error = pdb_wc.run(f, None)
            logger.info(f"Has error: {'Yes' if has_error else 'No'}")
    if pdb_path.suffix == ".cif":
        with pdb_path.open("r") as f:
            complex_pdb_str = run_pdbtools(f, pdb_fromcif)
    else:
        complex_pdb_str = pdb_path.read_text()

    complex_pdb_str = run_pdbtools(StringIO(complex_pdb_str), pdb_delelem, "H")
    save_complex_path = pdb_path.parent / "complex_fixed.pdb"
    if save_complex_path.exists() and not force_write:
        logger.info(f"File {save_complex_path} already exists")
    else:
        fix_pdb(StringIO(complex_pdb_str), save_complex_path)

    split_structure(
        save_complex_path,
        peptide_chains,
        protein_chains,
        force_split=force_write,
    )


def replace_and_record(input_text):
    try:
        matches = re.finditer(r"\((\w{3})\)", input_text)
        output_text = input_text
        offset = 0

        positions_list = []
        types_list = []
        for i, match in enumerate(matches, start=1):
            original_text = match.group(0)
            content = match.group(1)
            start_index = match.start()

            replace_position = start_index - offset
            offset += len(original_text) - 1  # 更新偏移量

            output_text = (
                output_text[:replace_position]
                + "?"
                + output_text[replace_position + len(original_text) :]
            )
            positions_list.append(replace_position + 1)
            types_list.append(content)
        return {
            "sequence": output_text,
            "has_ncaa": len(types_list) > 0,
            "positions": positions_list,
            "types": types_list,
            "length": len(output_text),
        }
    except Exception as e:
        raise RuntimeError(f"Error replacing and recording: {e}")


def extract_release_date_with_biopython(pdb_path: Path):
    structure = MMCIFParser(QUIET=True).get_structure(
        "protein", str(pdb_path.resolve())
    )
    return structure.header.get("deposition_date")


def get_job_info(cif_path: Path, row_info: dict):
    try:
        cif_doc = gemmi.cif.read_file(str(cif_path.resolve()))
        block = cif_doc.sole_block()  # 获取文件中的主要数据块
        job_info = {}
        # 查找 _entity_poly 的相关信息
        strand_id_column = block.find_values("_entity_poly.pdbx_strand_id")
        sequence_column = block.find_values("_entity_poly.pdbx_seq_one_letter_code")

        # 确保两列数据都存在
        if strand_id_column and sequence_column:
            for strand_ids, sequence in zip(strand_id_column, sequence_column):
                # 清理序列数据
                sequence = (
                    sequence.replace("\n", "")
                    .replace(" ", "")
                    .replace(";", "")
                    .replace("'", "")
                    .replace('"', "")
                )  # 清理换行和空格
                chains = strand_ids.split(",")

                # 遍历链 ID
                for chain in chains:
                    chain = chain.strip()
                    _chain_info = {"id": chain}
                    _chain_info.update(replace_and_record(sequence))
                    job_info[chain] = _chain_info

        job_info.update(row_info)
        job_info.update({"pdb_date": extract_release_date_with_biopython(cif_path)})
        return job_info
    except Exception as e:
        raise RuntimeError(f"Error getting job info: {e}")
