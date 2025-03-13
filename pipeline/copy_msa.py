from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
from loguru import logger

import sys

logger.remove()
logger.add(sys.stdout, level="INFO")


MODEL_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [
    89538,
    69825,
    92430,
    77417,
    88,
    76303,
    38060,
    45215,
    79677,
    25107,
    5942,
    68548,
    91320,
    36878,
    43339,
    23270,
    69773,
    73465,
    89579,
    73443,
    95522,
    77804,
    29712,
    24957,
    12029,
    16724,
    99622,
    56620,
    54183,
    28911,
    94231,
    45235,
    21326,
    2943,
    37634,
    87645,
    41036,
    69499,
    61139,
    42,
    59118,
    16599,
    33568,
    16448,
    80781,
    88875,
    92948,
    40691,
    79229,
    19452,
    77206,
    91791,
    399,
    25882,
    95944,
    76322,
    36574,
    7169,
    66666,
    15509,
    49861,
    86175,
    66624,
    43777,
    20319,
    9334,
    47664,
    84230,
    73130,
    72599,
    52547,
    18781,
    96734,
    9942,
    64730,
    56748,
    4131,
    74266,
    19270,
    22908,
    89066,
    88687,
    1710,
    72423,
    52399,
    69269,
    1314,
    75137,
    71267,
    73492,
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
