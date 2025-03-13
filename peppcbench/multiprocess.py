import multiprocessing
from pathlib import Path
import traceback
from gpustat import GPUStatCollection
from loguru import logger
from tqdm import tqdm
import asyncio
from queue import Queue, Full
from typing import List, Optional


class OrderedTaskProcessor:
    def __init__(self, task_function, num_workers=4, timeout=300):
        self.task_function = task_function
        self.num_workers = num_workers
        self.timeout = timeout

    def execute_tasks(self, task_args):
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = []
            for arg in task_args:
                result = pool.apply_async(self.task_function, args=(arg,))
                results.append(result)
            # 收集结果，保持顺序
            ordered_results = []
            for result in tqdm(results, total=len(results), desc="Processing Tasks"):
                try:
                    res = result.get(timeout=self.timeout)
                    if res is None:
                        continue
                    ordered_results.append(res)
                except multiprocessing.TimeoutError:
                    logger.error(f"Task timeout, timeout is {self.timeout} seconds.")
                    continue
                except Exception as e:
                    raise Exception(
                        f"Task execution failed: {e}\n{traceback.format_exc()}"
                    )
        return ordered_results

    def _wrapper_task(self, task_arg):
        self.task_function(task_arg)


def get_status(status_dir: Path, mode: str) -> str:
    """
    检查指定模式 (msa/infer) 的状态。

    :param status_dir: 状态文件所在目录
    :param mode: 模式 ("msa" 或 "infer")
    :return: 状态 ("running", "done", "error" 或 "None")
    """
    running_file = status_dir / f"{mode}.running"
    done_file = status_dir / f"{mode}.done"
    error_file = status_dir / f"{mode}.error"

    if running_file.exists():
        return "running"
    elif done_file.exists():
        return "done"
    elif error_file.exists():
        return "error"
    else:
        return "None"


class GPUQueue:
    def __init__(self, gpus: List[int]):
        self.gpus = gpus
        self.max_queue_size = len(self.gpus)
        self.queue = Queue(maxsize=self.max_queue_size)
        self._initialize_queue()

    def _initialize_queue(self):
        """初始化队列，将所有 GPU 加入池中"""
        for gpu in self.gpus:
            self.queue.put(gpu)

    async def acquire_gpu(self, min_free_memory: int = 50 * 1024) -> Optional[int]:
        """
        异步获取一个满足条件的 GPU
        :param min_free_memory: 最小空闲内存（单位：MB）
        :return: 满足条件的 GPU ID 或 None
        """
        while True:
            try:
                # 尝试从队列中获取 GPU
                gpu_id = self.queue.get_nowait()

                # 检查该 GPU 的空闲内存是否满足条件
                if self._is_gpu_suitable(gpu_id, min_free_memory):
                    print(f"Acquired GPU {gpu_id} with sufficient memory.")
                    return gpu_id
                else:
                    # 不满足条件，将 GPU 重新放回队列
                    self.release_gpu(gpu_id)
                    await asyncio.sleep(
                        30
                    )  # 等待 30 秒再尝试获取  每次 5 min / 4 个 GPU

            except Full:
                # 如果队列为空，等待 GPU 资源释放
                print("All GPUs are currently in use, waiting...")
                await asyncio.sleep(30)  # 等待 30 秒再尝试获取  每次 5 min / 4 个 GPU

    def release_gpu(self, gpu: int):
        """释放一个 GPU"""
        self.queue.put(gpu)
        print(f"Released GPU {gpu} back to the pool.")

    @staticmethod
    def _is_gpu_suitable(gpu_id: int, min_free_memory: int) -> bool:
        """
        检查 GPU 是否有足够的空闲内存
        :param gpu_id: GPU ID
        :param min_free_memory: 最小空闲内存要求
        :return: 是否满足要求
        """
        try:
            gpu_stats = GPUStatCollection.new_query()
            gpu_info = gpu_stats[gpu_id]
            free_memory = gpu_info.memory_free  # 单位是 MB

            if free_memory >= min_free_memory:
                print(f"GPU {gpu_id} has {free_memory} MB free memory.")
                return True
            else:
                print(f"GPU {gpu_id} has insufficient memory: {free_memory} MB.")
                return False
        except Exception as e:
            print(f"Error while checking GPU {gpu_id}: {e}")
            return False
