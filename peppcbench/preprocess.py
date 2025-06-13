import json
import os
from pathlib import Path
from io import StringIO
import re
import subprocess
import gemmi
from loguru import logger
from typing import Any, List
import numpy as np
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
import openmm.app as app
from DockQ.DockQ import load_PDB
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB import PDBIO, PDBList
from Bio.PDB.Residue import Residue
from scipy.optimize import linear_sum_assignment
from peppcbench.utils import dict2json
import time


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
    suffix_name: str = None,
):
    """
    Example inputs: 
    structure_path = "/data/home/silong/paper/PepPCBench/data/pdb/1a0q/1a0q.cif"
    peptide_chains = "A:"
    protein_chains = "B:C:D"
    """
    structure_path = structure_path.resolve()
    peptide_save_path = structure_path.parent / "peptide.pdb"
    pocket_save_path = structure_path.parent / "pocket.pdb"
    complex_save_path = structure_path.parent / "complex.pdb"
    protein_save_path = structure_path.parent / "protein.pdb"
    peptide_pocket_save_path = structure_path.parent / "peptide_pocket.pdb"
    
    if suffix_name:
        peptide_save_path = structure_path.parent / f"peptide_{suffix_name}.pdb"
        pocket_save_path = structure_path.parent / f"pocket_{suffix_name}.pdb"
        complex_save_path = structure_path.parent / f"complex_{suffix_name}.pdb"
        protein_save_path = structure_path.parent / f"protein_{suffix_name}.pdb"
        peptide_pocket_save_path = structure_path.parent / f"peptide_pocket_{suffix_name}.pdb"

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
        result = subprocess.run(maxit_cmd, check=True)
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
    save_complex_path = pdb_path.parent / "complex_fixed.pdb"
    if save_complex_path.exists() and not force_write:
        logger.info(f"File {save_complex_path} already exists")
    else:
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


def process_pdb_chains(
    pdb_model: Model,
    rec_chains: List[str],
    pep_chain: str,
    save_path: Path = None,
) -> str:

    # 创建新结构
    new_structure = Structure(0)
    new_model = Model(0)

    # 处理肽链
    pep_chain_new = Chain("A")
    pep_residues_seen = set()  # 用于跟踪已处理的残基编号
    for chain in pdb_model:
        if chain.id == pep_chain:
            for residue in chain:
                # 跳过已经处理过的残基编号
                res_id = residue.get_id()
                if res_id in pep_residues_seen:
                    continue
                pep_residues_seen.add(res_id)
                pep_chain_new.add(residue.copy())

    # 处理受体链
    rec_chain_new = Chain("B")
    rec_residue_counter = 1  # 初始化受体链残基编号
    for chain in pdb_model:
        if chain.id in rec_chains:
            for residue in chain:
                if residue.id[0] != " ":  # 跳过非标准残基（如水分子）
                    continue
                # 创建新的残基ID，(hetfield, resseq, icode)
                new_res_id = (" ", rec_residue_counter, " ")
                rec_residue_counter += 1
                # 创建新的Residue对象
                new_residue = Residue(new_res_id, residue.resname, residue.segid)
                # 复制所有原子
                for atom in residue:
                    new_residue.add(atom.copy())  # 添加原子到新残基
                rec_chain_new.add(new_residue)

    # 添加链到模型
    new_model.add(pep_chain_new)
    new_model.add(rec_chain_new)
    new_structure.add(new_model)

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(save_path.open("wt"))
    logger.debug(f"肽链 ({pep_chain} -> A) 残基数: {len(pep_chain_new)}")
    logger.debug(f"受体链 ({','.join(rec_chains)} -> B) 残基数: {len(rec_chain_new)}")


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # 删除
                dp[i][j - 1] + 1,  # 插入
                dp[i - 1][j - 1] + cost,  # 替换
            )
    return dp[m][n]


def sequence_similarity(s1, s2):
    dist = edit_distance(s1, s2)
    length = max(len(s1), len(s2))
    similarity = 1 - dist / length
    return max(similarity, 0)


def fuzzy_mapping(seq1_dict, seq2_dict):
    """
    seq1_dict: {A: 'XXXXX', B: 'XXXXX', C: 'XXXXX', D: 'XXXXX'} 大写字母为键
    seq2_dict: {a: 'xxxxx', b: 'xxxxx', c: 'xxxxx', d: 'xxxxx'} 小写字母为键

    将seq1_dict中每个序列与seq2_dict中的序列计算相似度，选出最匹配且唯一的seq2键。
    返回一个字典，如 {A: 'b', B: 'c', C: 'd', D: 'a'}。
    """
    # 提取键列表
    keys_seq1 = list(seq1_dict.keys())
    keys_seq2 = list(seq2_dict.keys())

    num_seq1 = len(keys_seq1)
    num_seq2 = len(keys_seq2)

    # 检查是否可以进行一对一映射
    if num_seq1 > num_seq2:
        raise ValueError("seq1_dict中的键数量超过了seq2_dict，无法进行一对一映射。")

    # 构建相似度矩阵
    similarity_matrix = np.zeros((num_seq1, num_seq2))
    for i, k1 in enumerate(keys_seq1):
        for j, k2 in enumerate(keys_seq2):
            similarity_matrix[i][j] = sequence_similarity(seq1_dict[k1], seq2_dict[k2])

    # 将相似度矩阵转换为成本矩阵
    # 因为linear_sum_assignment是最小化成本，所以我们取最大相似度的负数作为成本
    cost_matrix = -similarity_matrix

    # 使用匈牙利算法找到最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 构建结果映射字典
    result = {}
    for i, j in zip(row_ind, col_ind):
        k1 = keys_seq1[i]
        k2 = keys_seq2[j]
        result[k1] = k2

    return result


def chain_map(pdb_model: Model, pdb_native: Model):
    """
    Map chains between predicted model and native structure based on sequence similarity.

    Parameters:
        pdb_model (Model): Predicted model structure.
        pdb_native (Model): Native structure.

    Returns:
        dict: Mapping from native chains to model chains.
    """
    model_chains = {
        chain.id: "".join(res.resname for res in chain.get_residues())
        for chain in pdb_model.get_chains()
    }
    native_chains = {
        chain.id: "".join(res.resname for res in chain.get_residues())
        for chain in pdb_native.get_chains()
    }

    # chain_mapping = {}
    # for model_chain_id, model_seq in model_chains.items():
    #     matched_native_chain_id = get_most_similar_seq(model_seq, native_chains)
    #     chain_mapping[matched_native_chain_id] = model_chain_id
    # return chain_mapping
    return fuzzy_mapping(native_chains, model_chains)


def process_pdb_pair_chains(
    pdb_path: Path,
    pdb_ref_path: Path,
    peptide_chains: str,
    protein_chains: str,
    force_write: bool = False,
    suffix_name: str = "complex_processed",
):
    processed_pdb_path = pdb_path.parent / f"{suffix_name}.pdb"
    processed_pdb_ref_path = pdb_ref_path.parent / f"{suffix_name}.pdb"

    if (
        processed_pdb_path.exists()
        and processed_pdb_ref_path.exists()
        and not force_write
    ):
        logger.debug(
            f"File {processed_pdb_path} and {processed_pdb_ref_path} already exists"
        )

    else:
        pdb_model: Structure = load_PDB(str(pdb_path.resolve()))

        peptide_chain = peptide_chains.split(":")[0]
        protein_chains = protein_chains.split(":")

        select_ref_chains = protein_chains + [peptide_chain]
        pdb_ref: Structure = load_PDB(
            str(pdb_ref_path.resolve()), chains=select_ref_chains
        )
        save_chain_mapping_path = processed_pdb_path.parent / "chain_mapping.json"
        if save_chain_mapping_path.exists() and not force_write:
            logger.debug(f"File {save_chain_mapping_path} already exists")
            chain_mapping = json.load(save_chain_mapping_path.open("r"))
        else:
            chain_mapping = chain_map(pdb_model, pdb_ref)
            save_chain_mapping_path.write_text(dict2json(chain_mapping))

        logger.debug(chain_mapping)

        process_pdb_chains(
            pdb_ref,
            protein_chains,
            peptide_chain,
            save_path=processed_pdb_ref_path,
        )

        process_pdb_chains(
            pdb_model,
            list(chain_mapping[chain] for chain in protein_chains),
            chain_mapping[peptide_chain],
            save_path=processed_pdb_path,
        )


def process_pdb_pair_chains_for_relax(
    pdb_path: Path,
    pdb_ref_path: Path,
    peptide_chains: str,
    protein_chains: str,
    force_write: bool = False,
    suffix_name: str = "complex_relaxed_processed",
):
    processed_pdb_path = pdb_path.parent / f"{suffix_name}.pdb"

    if processed_pdb_path.exists() and not force_write:
        logger.debug(f"File {processed_pdb_path} already exists")
    else:
        pdb_model: Structure = load_PDB(str(pdb_path.resolve()))
        peptide_chain = peptide_chains.split(":")[0]
        protein_chains = protein_chains.split(":")

        select_ref_chains = protein_chains + [peptide_chain]
        pdb_ref: Structure = load_PDB(
            str(pdb_ref_path.resolve()), chains=select_ref_chains
        )
        save_chain_mapping_path = (
            processed_pdb_path.parent / "chain_mapping_relaxed.json"
        )
        if save_chain_mapping_path.exists() and not force_write:
            logger.debug(f"File {save_chain_mapping_path} already exists")
            chain_mapping = json.load(save_chain_mapping_path.open("r"))
        else:
            chain_mapping = chain_map(pdb_model, pdb_ref)
            save_chain_mapping_path.write_text(dict2json(chain_mapping))

        logger.debug(chain_mapping)

        process_pdb_chains(
            pdb_model,
            list(chain_mapping[chain] for chain in protein_chains),
            chain_mapping[peptide_chain],
            save_path=processed_pdb_path,
        )
