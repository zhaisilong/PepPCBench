# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Post-processing utilities for AlphaFold inference results."""

import dataclasses
import datetime
import os

from alphafold3 import version
from alphafold3.model import confidence_types
from alphafold3.model import mmcif_metadata
from alphafold3.model import model
import numpy as np
import pickle


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProcessedInferenceResult:
  """Stores attributes of a processed inference result.

  Attributes:
    cif: CIF file containing an inference result.
    mean_confidence_1d: Mean 1D confidence calculated from confidence_1d.
    ranking_score: Ranking score extracted from CIF metadata.
    structure_confidence_summary_json: Content of JSON file with structure
      confidences summary calculated from CIF file.
    structure_full_data_json: Content of JSON file with structure full
      confidences calculated from CIF file.
    model_id: Identifier of the model that produced the inference result.
  """

  cif: bytes
  mean_confidence_1d: float
  ranking_score: float
  structure_confidence_summary_json: bytes
  structure_full_data_json: bytes
  model_id: bytes
  extra_output: bytes

def post_process_inference_result(
    inference_result: model.InferenceResult,
) -> ProcessedInferenceResult:
  """Returns cif, confidence_1d_json, confidence_2d_json, mean_confidence_1d, and ranking confidence."""

  # Add mmCIF metadata fields.
  timestamp = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
  cif_with_metadata = mmcif_metadata.add_metadata_to_mmcif(
      old_cif=inference_result.predicted_structure.to_mmcif_dict(),
      version=f'{version.__version__} @ {timestamp}',
      model_id=inference_result.model_id,
  )
  cif = mmcif_metadata.add_legal_comment(cif_with_metadata.to_string())
  cif = cif.encode('utf-8')
  confidence_1d = confidence_types.AtomConfidence.from_inference_result(
      inference_result
  )
  mean_confidence_1d = np.mean(confidence_1d.confidence)
  structure_confidence_summary_json = (
      confidence_types.StructureConfidenceSummary.from_inference_result(
          inference_result
      )
      .to_json()
      .encode('utf-8')
  )
  structure_full_data_json = (
      confidence_types.StructureConfidenceFull.from_inference_result(
          inference_result
      )
      .to_json()
      .encode('utf-8')
  )
  extra_output = (
      confidence_types.StructureConfidenceExtra.from_inference_result(
          inference_result
      )
      .to_pickle()
  )
  return ProcessedInferenceResult(
      cif=cif,
      mean_confidence_1d=mean_confidence_1d,
      ranking_score=float(inference_result.metadata['ranking_score']),
      structure_confidence_summary_json=structure_confidence_summary_json,
      structure_full_data_json=structure_full_data_json,
      model_id=inference_result.model_id,
      extra_output=extra_output,
  )


def write_output(
    inference_result: model.InferenceResult,
    output_dir: os.PathLike[str] | str,
    terms_of_use: str | None = None,
    name: str | None = None,
) -> None:
  """Writes processed inference result to a directory."""
  processed_result = post_process_inference_result(inference_result)

  prefix = f'{name}_' if name is not None else ''

  with open(os.path.join(output_dir, f'{prefix}model.cif'), 'wb') as f:
    f.write(processed_result.cif)

  with open(
      os.path.join(output_dir, f'{prefix}summary_confidences.json'), 'wb'
  ) as f:
    f.write(processed_result.structure_confidence_summary_json)

  with open(os.path.join(output_dir, f'{prefix}confidences.json'), 'wb') as f:
    f.write(processed_result.structure_full_data_json)

  if terms_of_use is not None:
    with open(os.path.join(output_dir, 'TERMS_OF_USE.md'), 'wt') as f:
      f.write(terms_of_use)
  
  with open(os.path.join(output_dir, f'{prefix}extra_output.pkl'), 'wb') as f:
    f.write(processed_result.extra_output)


def write_embeddings(
    embeddings: dict[str, np.ndarray],
    output_dir: os.PathLike[str] | str,
) -> None:
  """Writes embeddings to a directory."""

  with open(os.path.join(output_dir, 'embeddings.npz'), 'wb') as f:
    np.savez_compressed(f, **embeddings)
