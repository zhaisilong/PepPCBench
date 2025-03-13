import itertools
import json
from loguru import logger
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import List
from peppcbench.multiprocess import GPUQueue
from peppcbench.multiprocess import get_status
from peppcbench import utils
from peppcbench.configs import AF3Config, ProteinConfig
from peppcbench.models import AF3
import matplotlib.pyplot as plt

import sys

logger.remove()
logger.add(sys.stdout, level="INFO")


class Pipeline:
    def __init__(self, data_dir: str = "pepdb"):
        self.data_dir = Path(data_dir)
        assert self.data_dir.exists(), f"Data directory not found: {self.data_dir}"
        self.job_list = [
            job_dir for job_dir in self.data_dir.iterdir() if job_dir.is_dir()
        ]

    def gen_config(self, model_name: str = "af3"):
        for job_dir in tqdm(self.job_list):
            job_info_path = job_dir / "job_info.json"
            assert job_info_path.exists(), f"Job info not found: {job_info_path}"
            job_info = json.loads(job_info_path.read_text())
            model_dir = job_dir / model_name
            model_dir.mkdir(exist_ok=True)

            config_path = model_dir / f"{job_info['job_name']}.json"

            protein_chains = job_info["protein_chains"].split(":")
            peptide_chains = job_info["peptide_chains"].split(":")
            sequences = []

            for chain in itertools.chain(peptide_chains, protein_chains):
                sequence = job_info[chain]["sequence"]
                modifications = None
                if job_info[chain]["has_ncaa"]:
                    modifications = []
                    for pos, ncaa_three in zip(
                        job_info[chain]["positions"], job_info[chain]["types"]
                    ):
                        pos = int(pos)
                        try:
                            sequence = utils.replace_char(
                                sequence,
                                pos - 1,
                                utils.letters_three_to_one(ncaa_three, default="X"),
                            )
                            modifications.append(
                                {"ptmType": ncaa_three, "ptmPosition": pos},
                            )
                        except Exception as e:
                            print(e)

                assert "?" not in sequence, f"{job_info['job_name']} has ? in sequence"

                sequences.append(
                    ProteinConfig(
                        id=chain,
                        sequence=sequence,
                        unpaired_msa=None,
                        paired_msa=None,
                        templates=None,
                        modifications=modifications,
                    )
                )

            af3_dict = {
                "name": job_info["job_name"],
                "modelSeeds": [42],
                "sequences": sequences,
                "dialect": "alphafold3",
                "version": 1,
            }
            af3_config = AF3Config.from_dict(af3_dict)
            utils.save_json(utils.dict2json(af3_config.to_dict()), config_path)

    def run_model(
        self, model_name: str = "af3", mode: str = "full", gpus: List = [0, 1, 2, 3]
    ):
        if model_name == "af3":
            self.run_af3(mode=mode, gpus=gpus)
        elif model_name == "afm":
            self.run_afm(mode=mode, gpus=gpus)
        elif model_name == "rfaa":
            self.run_rfaa(mode=mode, gpus=gpus)
        elif model_name == "chai":
            self.run_chai(mode=mode, gpus=gpus)
        elif model_name == "helix":
            self.run_helix(mode=mode, gpus=gpus)
        else:
            raise ValueError(f"Model {model_name} not supported")

    def run_af3(self, mode: str = "full", gpus: List = [0, 1, 2, 3]):
        assert len(gpus) > 0, "No GPUs provided"
        assert self.job_list, "No jobs to run"
        af3_model = AF3(
            bash_path=Path(
                "/data/home/silong/paper/PepPCBench/scripts/run_alphafold.sh"
            ),
            gpu_pool=GPUQueue(gpus=gpus),
        )

        for job_dir in tqdm(self.job_list, desc="Running AF3 jobs"):
            job_info_path = job_dir / "job_info.json"
            job_info = json.loads(job_info_path.read_text())
            job_model_dir = job_dir / "af3"

            af3_model(
                job_name=job_info["job_name"],
                model_job_dir=Path(job_model_dir),
                mode=mode,
                verbose=True,
            )

    def summary_jobs(
        self,
        output_csv: str = "summary_jobs.csv",
        model_name_list: List[str] = ["af3", "rfaa", "chai", "helix", "afm"],
        num_jobs: int = -1,
        gen_fig: bool = False,
    ):
        """
        汇总任务信息并写入 CSV 文件。

        :param output_csv: 输出的 CSV 文件路径
        :param verbose: 是否打印调试信息
        """
        assert self.jobs, "No jobs to summarize."
        output_csv = Path(output_csv)
        select_cols = [
            "job_name",
            "job_dir",
            "pdb_id",
            "pdb_date",
            "protein_chains",
            "peptide_chains",
        ]
        all_infos = []  # 用于存储每个 job 的 info 数据

        for job in tqdm(self.jobs[:num_jobs], desc="Processing jobs"):
            info = job.info  # 获取 job 的信息字典
            # logger.info(f"job.info: {info}")
            assert isinstance(
                info, dict
            ), f"job.info() must return a dict, got {type(info)}"
            peptide_chain_id = info.get("peptide_chains", None)
            peptide_chain_len = info.get(peptide_chain_id, {}).get("length", 0)
            peptide_chain_has_ncaa = info.get(peptide_chain_id, {}).get(
                "has_ncaa", False
            )

            # 筛选需要的列
            filtered_info = {col: info.get(col, None) for col in select_cols}
            filtered_info.update(
                {
                    "peptide_chain_len": peptide_chain_len,
                    "peptide_chain_has_ncaa": peptide_chain_has_ncaa,
                }
            )
            job_dir = Path(info.get("job_dir", ""))
            assert job_dir
            if not model_name_list:
                model_name_list = ["af3"]

            for model_name in model_name_list:
                status_dir = job_dir / model_name

                # 检查 MSA 和 Infer 状态
                msa_status = get_status(status_dir, "msa")
                infer_status = get_status(status_dir, "infer")

                # 添加状态信息
                filtered_info[f"msa_status_{model_name}"] = msa_status
                filtered_info[f"infer_status_{model_name}"] = infer_status

            all_infos.append(filtered_info)

        df = pd.DataFrame(all_infos)
        df.to_csv(output_csv, index=False)

        # 生成实验进度条
        if gen_fig:
            statuses = [
                f"{status}_{model}"
                for model in model_name_list
                for status in ("msa_status", "infer_status")
            ]
            # Count the number of "done" and "not done" statuses for each type
            status_counts = {
                status: {
                    "done": df[status].apply(lambda x: x == "done").sum(),
                    "not_done": df[status].apply(lambda x: x != "done").sum(),
                }
                for status in statuses
            }

            # Prepare data for plotting
            done_counts = [status_counts[status]["done"] for status in statuses]
            not_done_counts = [status_counts[status]["not_done"] for status in statuses]

            x_labels = statuses
            x = range(len(x_labels))

            # Plot the bar chart
            fig, ax = plt.subplots(dpi=300)

            # Plotting "done" as green bars
            ax.bar(x, done_counts, label="Done", color="green")

            # Plotting "not done" as gray bars stacked above
            ax.bar(x, not_done_counts, label="Undone", color="gray", bottom=done_counts)

            # Customize the plot
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, fontsize=10)
            ax.set_xlabel("Status Type")
            ax.set_ylabel("Number of Jobs")
            max_jobs = max(done_counts + not_done_counts)
            y_limit = int(max_jobs * 1.5)  # Add 10%
            # Apply the y-axis limit
            ax.set_ylim(0, y_limit)
            ax.legend()

            plt.tight_layout()

            # Save the plot
            output_path = str(output_csv.resolve()).replace(".csv", ".png")
            plt.savefig(output_path)
