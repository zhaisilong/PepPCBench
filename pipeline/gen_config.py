from peppcbench.pipeline import Pipeline
import fire


def main(data_dir: str = "./pepdb", model_name: str = "af3"):
    pipeline = Pipeline(data_dir=data_dir)
    pipeline.gen_config(model_name=model_name)


if __name__ == "__main__":
    fire.Fire(main)
