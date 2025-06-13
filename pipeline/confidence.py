import json
from pathlib import Path
import pickle
from typing import Tuple
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO")


BREAKS = np.linspace(0.0, 31.0, 63)
NUM_DIGITS = 4


def _calculate_bin_centers(breaks):
    """Gets the bin centers from the bin edges.
    Args:
      breaks: [num_bins - 1] the error bin edges.
    Returns:
      bin_centers: [num_bins] the error bin centers.
    """
    step = breaks[1] - breaks[0]

    # Add half-step to get the center
    bin_centers = breaks + step / 2

    # Add a catch-all bin at the end.
    return np.append(bin_centers, bin_centers[-1] + step)


def cal_actifptm(pae_probs, cmap, asym_id, start_i, end_i, start_j, end_j):
    total_length = len(asym_id)
    cmap_copy = np.zeros((total_length, total_length))
    cmap_copy[start_i : end_i + 1, start_j : end_j + 1] = cmap[
        start_i : end_i + 1, start_j : end_j + 1
    ]
    cmap_copy[start_j : end_j + 1, start_i : end_i + 1] = cmap[
        start_j : end_j + 1, start_i : end_i + 1
    ]
    pair_mask = asym_id[:, None] != asym_id[None, :]

    # this is for the full-length actifptm
    if end_i == end_j == total_length - 1 and start_i == start_j == 0:
        cmap_copy *= pair_mask

    seq_mask = np.full(total_length, 0, dtype=float)
    seq_mask[
        np.concatenate((np.arange(start_i, end_i + 1), np.arange(start_j, end_j + 1)))
    ] = 1
    clipped_num_res = np.maximum(seq_mask.sum(), 19)
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
    bin_centers = _calculate_bin_centers(BREAKS)
    tm_per_bin = 1.0 / (1 + np.square(bin_centers) / np.square(d0))

    predicted_tm_term = (pae_probs * tm_per_bin).sum(-1)
    predicted_tm_term *= pair_mask

    normed_residue_mask = cmap_copy / (1e-8 + cmap_copy.sum(-1, keepdims=True))

    per_alignment = (predicted_tm_term * normed_residue_mask).sum(-1)
    residuewise_iptm = per_alignment * seq_mask
    return residuewise_iptm


def cal_iptm(pae_probs, asym_id, start_i, end_i, start_j, end_j):
    residue_weights = np.full(len(asym_id), 0, dtype=float)
    residue_weights[
        np.concatenate((np.arange(start_i, end_i + 1), np.arange(start_j, end_j + 1)))
    ] = 1
    bin_centers = _calculate_bin_centers(BREAKS)
    num_res = residue_weights.shape[0]
    clipped_num_res = np.maximum(residue_weights.sum(), 19)
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
    tm_per_bin = 1.0 / (1 + np.square(bin_centers) / np.square(d0))
    predicted_tm_term = (pae_probs * tm_per_bin).sum(-1)

    pair_mask = asym_id[:, None] != asym_id[None, :]
    predicted_tm_term *= pair_mask
    pair_residue_weights = pair_mask * (
        residue_weights[None, :] * residue_weights[:, None]
    )

    normed_residue_mask = pair_residue_weights / (
        1e-8 + pair_residue_weights.sum(-1, keepdims=True)
    )
    per_alignment = (predicted_tm_term * normed_residue_mask).sum(-1)
    residuewise_iptm = per_alignment * residue_weights
    return residuewise_iptm


def cal_cptm(pae_probs, start, end):
    pae_probs = pae_probs[start : end + 1, start : end + 1, :]
    residue_weights = np.ones(pae_probs.shape[0])
    num_res = residue_weights.shape[0]
    clipped_num_res = np.maximum(residue_weights.sum(), 19)
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
    bin_centers = _calculate_bin_centers(BREAKS)
    tm_per_bin = 1.0 / (1 + np.square(bin_centers) / np.square(d0))
    predicted_tm_term = (pae_probs * tm_per_bin).sum(-1)

    pair_mask = np.full((num_res, num_res), True)
    predicted_tm_term *= pair_mask
    pair_residue_weights = pair_mask * (
        residue_weights[None, :] * residue_weights[:, None]
    )
    normed_residue_mask = pair_residue_weights / (
        1e-8 + pair_residue_weights.sum(-1, keepdims=True)
    )
    per_alignment = (predicted_tm_term * normed_residue_mask).sum(-1)
    residuewise_iptm = per_alignment * residue_weights

    return residuewise_iptm


def cal_ipae(pae_probs, start_i, end_i, start_j, end_j, method="mean"):
    """
    计算跨链(Inter-Chain) PAE：对两个子矩阵(上三角/下三角)取并集后，
    按照指定 method（mean 或 min）返回 PAE 值。

    :param pae_probs: (N, N, 63) 或 (N, N, bin_count) 的 PAE 概率分布
    :param start_i, end_i: 链 i 在全序列中的起止索引
    :param start_j, end_j: 链 j 在全序列中的起止索引
    :param method: 计算方式，可选 "mean" 或 "min"
    :return: float, 对应子矩阵的 PAE 值
    """
    # 将日志阈值范围在[0,31],共63 bins (与AlphaFold一致)
    bin_centers = _calculate_bin_centers(BREAKS)
    
    # 先将 prob × bin_centers 求和得到期望 PAE 矩阵
    expected_pae = np.sum(pae_probs * bin_centers, axis=-1)  # shape: [N, N]

    # 提取跨链的两个子矩阵(对称)
    sub_pae_block_1 = expected_pae[start_i : end_i + 1, start_j : end_j + 1]
    sub_pae_block_2 = expected_pae[start_j : end_j + 1, start_i : end_i + 1]
    
    # 将这两个子矩阵的所有值合并
    combined_values = np.concatenate([sub_pae_block_1.flatten(), sub_pae_block_2.flatten()])
    
    if method == "mean":
        return float(combined_values.mean())
    elif method == "min":
        return float(combined_values.min())
    else:
        raise ValueError(f"Invalid method: {method}")


def cal_cpae(pae_probs, start, end, method="mean"):
    """
    计算单链(Chain) PAE：对链内的子矩阵进行统计(均值或最小值)

    :param pae_probs: (N, N, 63) 或 (N, N, bin_count) 的 PAE 概率分布
    :param start, end: 链在全序列中的起止索引
    :param method: 计算方式，可选 "mean" 或 "min"
    :return: float, 对应子矩阵的 PAE 值
    """
    bin_centers = _calculate_bin_centers(BREAKS)
    expected_pae = np.sum(pae_probs * bin_centers, axis=-1)  # shape: [N, N]
    sub_pae = expected_pae[start : end + 1, start : end + 1]
    
    if method == "mean":
        return float(sub_pae.mean())
    elif method == "min":
        return float(sub_pae.min())
    else:
        raise ValueError(f"Invalid method: {method}")


def get_contact_map(dist_logits, dist_cutoff=8.0):
    dist_bins = np.append(0, np.linspace(2.3125, 21.6875, 63))
    dist_probs = softmax(dist_logits, axis=-1)
    bin_centers = _calculate_bin_centers(np.linspace(2.3125, 21.6875, 63))
    dist = np.sum(dist_probs * bin_centers, axis=-1)
    contact_map = np.sum(dist_probs * (dist_bins < dist_cutoff), axis=-1)
    return contact_map, dist


def get_chain_indices(asym_id):
    chain_starts_ends = []
    unique_chains = np.unique(asym_id)

    for chain in unique_chains:
        positions = np.where(asym_id == chain)[0]
        chain_starts_ends.append((positions[0], positions[-1]))

    return chain_starts_ends


def get_confidence(model_dir: Path) -> Tuple[float, float, float]:
    confidence_path = model_dir / "summary_confidences.json"
    assert confidence_path.exists(), f"{confidence_path} does not exist"
    confidence_content = json.loads(confidence_path.read_text())
    output = {}
    output["iptm"] = confidence_content["iptm"]
    output["ptm"] = confidence_content["ptm"]
    output["ranking_score"] = confidence_content["ranking_score"]
    return output


def get_confidence_extra(sample_dir: Path, force_write=False, if_pae=False):
    confidence_extra_path = sample_dir / "confidences_extra.json"
    selected_cols = [
        "iptm",
        "ptm",
        "ranking_score",
        "original_iptm",
        "original_ptm",
        "original_confidence_score",
        "confidence_score",
        "original_ranking_score",
        "actifptm",
    ]
    if if_pae:  # todo: 需要修改
        selected_cols.extend(["ipae", "ipae_min", "cpae", "cpae_min"])
    if confidence_extra_path.exists() and not force_write:
        return {
            k: v
            for k, v in json.loads(confidence_extra_path.read_text()).items()
            if k in selected_cols
        }

    extra_output_path = sample_dir / "extra_output.pkl"
    confidences_path = sample_dir / "confidences.json"
    summary_confidences_path = sample_dir / "summary_confidences.json"

    extra_output = pickle.loads(extra_output_path.read_bytes())
    confidences = json.loads(confidences_path.read_text())
    summary_confidences = json.loads(summary_confidences_path.read_text())

    asym_id, chain_labels = pd.factorize(np.array(confidences["token_chain_ids"]))
    full_length = len(asym_id)
    logger.debug(
        f"full_length: {full_length}, with {len(chain_labels)} chains: {chain_labels}"
    )

    pae_probs = softmax(extra_output["pae_logits"], axis=-1)
    contact_map = extra_output["contact_probs"]

    logger.debug(f"pae_probs.shape: {pae_probs.shape}")
    logger.debug(f"contact_map.shape: {contact_map.shape}")

    chain_starts_ends = get_chain_indices(asym_id)
    output = {
        "pairwise_actifptm": {},
        "pairwise_iptm": {},
        "per_chain_ptm": {},
        "fraction_disordered": summary_confidences["fraction_disordered"],
        "has_clash": summary_confidences["has_clash"],
        "original_iptm": summary_confidences["iptm"],
        "original_ptm": summary_confidences["ptm"],
        "ipae": {},
        "ipae_min": {},
        "cpae": {},
        "cpae_min": {},
    }
    for i, (start_i, end_i) in enumerate(chain_starts_ends):
        chain_label_i = chain_labels[
            i % len(chain_labels)
        ]  # Wrap around if more than 26 chains
        for j, (start_j, end_j) in enumerate(chain_starts_ends):
            chain_label_j = chain_labels[
                j % len(chain_labels)
            ]  # Wrap around if more than 26 chains
            if i < j:  # Avoid self-comparison and duplicate comparisons
                key = f"{chain_label_i}-{chain_label_j}"
                residuewise_actifptm = cal_actifptm(
                    pae_probs=pae_probs,
                    cmap=contact_map,
                    asym_id=asym_id,
                    start_i=start_i,
                    end_i=end_i,
                    start_j=start_j,
                    end_j=end_j,
                )
                pairwise_actifptm = round(float(residuewise_actifptm.max()), NUM_DIGITS)
                output["pairwise_actifptm"][key] = pairwise_actifptm
                # Also add regular i_ptm (interchain), pairwise_iptm
                residuewise_iptm = cal_iptm(
                    pae_probs=pae_probs,
                    asym_id=asym_id,
                    start_i=start_i,
                    end_i=end_i,
                    start_j=start_j,
                    end_j=end_j,
                )
                pairwise_iptm = round(float(residuewise_iptm.max()), NUM_DIGITS)
                output["pairwise_iptm"][key] = pairwise_iptm
                ipae = cal_ipae(
                    pae_probs=pae_probs,
                    start_i=start_i,
                    end_i=end_i,
                    start_j=start_j,
                    end_j=end_j,
                )
                ipae_min = cal_ipae(
                    pae_probs=pae_probs,
                    start_i=start_i,
                    end_i=end_i,
                    start_j=start_j,
                    end_j=end_j,
                    method="min",
                )
                output["ipae"][key] = round(ipae, NUM_DIGITS)
                output["ipae_min"][key] = round(ipae_min, NUM_DIGITS)
        # Also calculate pTM score for single chain
        cptm = cal_cptm(pae_probs=pae_probs, start=start_i, end=end_i)
        cptm = round(float(cptm.max()), NUM_DIGITS)
        output["per_chain_ptm"][chain_label_i] = cptm
        cpae = cal_cpae(pae_probs=pae_probs, start=start_i, end=end_i)
        cpae_min = cal_cpae(pae_probs=pae_probs, start=start_i, end=end_i, method="min")
        output["cpae"][chain_label_i] = round(cpae, NUM_DIGITS)
        output["cpae_min"][chain_label_i] = round(cpae_min, NUM_DIGITS)

    actifptm = round(
        float(
            cal_actifptm(
                pae_probs=pae_probs,
                cmap=contact_map,
                asym_id=asym_id,
                start_i=0,
                end_i=full_length - 1,
                start_j=0,
                end_j=full_length - 1,
            ).max()
        ),
        NUM_DIGITS,
    )
    output["actifptm"] = actifptm

    iptm = round(
        float(
            cal_iptm(
                pae_probs=pae_probs,
                asym_id=asym_id,
                start_i=0,
                end_i=full_length - 1,
                start_j=0,
                end_j=full_length - 1,
            ).max()
        ),
        NUM_DIGITS,
    )
    ptm = round(
        float(cal_cptm(pae_probs=pae_probs, start=0, end=full_length - 1).max()),
        NUM_DIGITS,
    )
    output["iptm"] = iptm
    output["ptm"] = ptm

    confidence_score = round(0.8 * actifptm + 0.2 * ptm, NUM_DIGITS)
    ranking_score = round(
        confidence_score
        + 0.5 * output["fraction_disordered"]
        - 100 * output["has_clash"],
        NUM_DIGITS,
    )
    original_confidence_score = round(0.8 * iptm + 0.2 * ptm, NUM_DIGITS)
    original_ranking_score = round(
        original_confidence_score
        + 0.5 * output["fraction_disordered"]
        - 100 * output["has_clash"],
        NUM_DIGITS,
    )

    output["confidence_score"] = confidence_score
    output["original_confidence_score"] = original_confidence_score
    output["ranking_score"] = ranking_score
    output["original_ranking_score"] = original_ranking_score
    json.dump(output, confidence_extra_path.open("w"), indent=2)
    return {k: v for k, v in output.items() if k in selected_cols}
