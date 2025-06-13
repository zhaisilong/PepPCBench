from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from loguru import logger

import sys

logger.remove()
logger.add(sys.stdout, level="INFO")


MODEL_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [
    42,
    777,
    25107,
    91791,
    34897,
    9999,
    100,
    52,
    2025,
    1000,
]


def replace_random_seed(config: dict, model_seeds: list) -> dict:
    config["modelSeeds"] = model_seeds
    return config


def update_config(from_path: Path, to_path: Path) -> None:
    with from_path.open("r", encoding="utf-8") as file:
        try:
            json_str = file.read()  # BUG 必须要用流，否则太大的数据无法读取
            config = json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {e}")
    config = replace_random_seed(config, MODEL_SEEDS)
    with to_path.open("w", encoding="utf-8") as file:
        try:
            json_str = json.dumps(config, indent=2, ensure_ascii=False)
            file.write(json_str)
        except Exception as e:
            raise ValueError(f"Failed to write JSON: {e}")


if __name__ == "__main__":
    source_dir = Path("/data/home/silong/projects/peptide/PepPCBench/work-3/pepdb")
    target_dir = Path("pepdb")
    target_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("job_list.csv")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        job_name = row["job_name"]
        source_job_dir = source_dir / job_name
        source_config_msa_path = (
            source_job_dir / "af3" / job_name.lower() / f"{job_name.lower()}_data.json"
        )
        source_config_path = source_job_dir / "af3" / f"{job_name}.json"

        target_job_dir = target_dir / job_name
        target_config_msa_dir = target_job_dir / "af3" / job_name.lower()
        target_config_msa_dir.mkdir(parents=True, exist_ok=True)
        target_config_msa_path = target_config_msa_dir / source_config_msa_path.name

        target_config_path = target_job_dir / "af3" / f"{job_name}.json"

        update_config(source_config_msa_path, target_config_msa_path)
        update_config(source_config_path, target_config_path)

        # Set msa status to done
        msa_status_path = target_job_dir / "af3" / "msa.done"
        msa_status_path.touch()
        logger.info(f"Successfully set msa status to done: {msa_status_path}")
