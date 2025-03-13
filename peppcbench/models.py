import asyncio
import subprocess
from pathlib import Path
import sys
from peppcbench.multiprocess import GPUQueue
import traceback


class BaseModel:
    def __init__(self, gpu_pool: GPUQueue, bash_path: str):
        self.gpu_pool = gpu_pool
        self.bash_path = bash_path

    def manage_status_files(self, model_job_dir, mode, status):
        """管理状态文件"""
        status_file = model_job_dir / f"{mode}.{status}"
        running_file = model_job_dir / f"{mode}.running"
        error_file = model_job_dir / f"{mode}.error"

        if status == "running":
            running_file.touch()
            print(f"Created running file: {running_file}")
        elif status == "done":
            if running_file.exists():
                running_file.unlink()
                print(f"Deleted running file: {running_file}")
            if error_file.exists():
                error_file.unlink()
                print(f"Deleted error file: {error_file}")
            status_file.touch()
            print(f"Created done file: {status_file}")

        elif status == "error":
            if running_file.exists():
                running_file.unlink()
                print(f"Deleted running file: {running_file}")
            error_file.touch()
            print(f"Created error file: {error_file}")

    def check_status(self, model_job_dir, mode):
        """检查状态文件"""
        done_file = model_job_dir / f"{mode}.done"
        running_files = [model_job_dir / "msa.running", model_job_dir / "infer.running"]
        error_file = model_job_dir / f"{mode}.error"

        msa_done = model_job_dir / "msa.done"

        # 检查是否有任何 .running 文件
        for running_file in running_files:
            if running_file.exists():
                print(
                    f"{mode}: A task is currently running ({running_file.name}). Skipping."
                )
                return "running"

        # 检查当前模式的 .done 文件
        if done_file.exists():
            print(f"{mode} already completed. Skipping.")
            return "done"

        if mode == "infer" and not msa_done.exists():
            print(f"{mode} requires msa first")
            return "error"

        # 检查当前模式的 .error 文件
        if error_file.exists():
            print(f"{mode} encountered an error previously. Retrying.")
            return "error"

        # 如果没有任何标记文件，任务为 pending
        return "pending"

    async def run_mode(self, job_name, model_job_dir, mode, verbose):
        """运行指定模式"""
        status = self.check_status(model_job_dir, mode)
        if status in ["done", "running"]:
            return  # 跳过已完成或正在运行的任务

        try:
            self.manage_status_files(model_job_dir, mode, "running")
            cmd = await self.generate_cmd(
                job_name, model_job_dir, mode == "msa", mode == "infer"
            )
            if verbose:
                print(f"Running command: {cmd}")
            subprocess.run(cmd, check=True)
            self.manage_status_files(model_job_dir, mode, "done")
            print(f"{mode} completed successfully.")

            if self.gpu_id is not None:  # 表示没分配过 就跳过
                self.gpu_pool.release_gpu(self.gpu_id)
                self.gpu_id = None

        except subprocess.CalledProcessError as e:
            self.manage_status_files(model_job_dir, mode, "error")
            print(f"Error: {mode} encountered an issue. {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            self.manage_status_files(model_job_dir, mode, "error")
            print(f"Run interrupted by user during {mode}.")
            sys.exit(1)
        except Exception as e:
            self.manage_status_files(model_job_dir, mode, "error")
            print(f"Error: {mode} encountered an unexpected issue. {e}")
            sys.exit(1)

    def __call__(
        self,
        job_name: str,
        model_job_dir: Path,
        mode: str = "msa",
        verbose: bool = True,
    ):

        if mode == "full":
            # full 模式：确保 msa 和 infer 都完成
            asyncio.run(self.run_mode(job_name, model_job_dir, "msa", verbose))
            asyncio.run(self.run_mode(job_name, model_job_dir, "infer", verbose))
        else:
            asyncio.run(self.run_mode(job_name, model_job_dir, mode, verbose))


class AF3(BaseModel):
    def __init__(
        self,
        gpu_pool: GPUQueue,
        bash_path: Path = Path(
            "/data/home/silong/paper/PepPCBench/scripts/run_alphafold.sh"
        ),
    ):
        super().__init__(gpu_pool, bash_path)

    async def generate_cmd(
        self,
        job_name: str,
        model_job_dir: Path,
        run_data_pipeline: bool,
        run_inference: bool,
    ):
        assert (
            run_data_pipeline or run_inference
        ), "At least one of run_data_pipeline or run_inference must be True"
        if run_inference:
            input_json_path = (
                model_job_dir / f"{job_name.lower()}/{job_name.lower()}_data.json"
            )
        else:
            input_json_path = model_job_dir / f"{job_name}.json"

        self.gpu_id = await self.gpu_pool.acquire_gpu(min_free_memory=80000)
        if self.gpu_id is not None:
            print(f"Using GPU {self.gpu_id} for computation.")

        cmd = [
            "bash",
            str(self.bash_path.resolve()),
            str(input_json_path.resolve()),
            str(model_job_dir.resolve()),
            str(run_data_pipeline).lower(),
            str(run_inference).lower(),
            str(self.gpu_id),
        ]

        return cmd
