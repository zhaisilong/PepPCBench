from peppcbench.pipeline import Pipeline
import fire


def main(data_dir: str = "./pepdb", model_name: str = "af3", gpu_id: int = 3):
    pipeline = Pipeline(data_dir=data_dir)
    pipeline.run_model(model_name=model_name, gpus=[gpu_id])


if __name__ == "__main__":
    fire.Fire(main)
