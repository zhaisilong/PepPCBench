from pathlib import Path
import pickle
import pandas as pd

def get_core_set(full_set: pd.DataFrame, clusters_test_path: Path = Path("../cluster2/clusters_test.pkl")):
    with clusters_test_path.open("br") as f:
        clusters_test = pickle.load(f)

    def get_redundency(job_name):
        pdb_id = job_name.split("_")[0]
        if pdb_id in clusters_test["test_independent_set"]:
            return 0
        elif pdb_id in clusters_test["test_redundent_set"]:
            return 1
        elif pdb_id in clusters_test["train_redundent_set"]:
            return 2
        else:
            raise ValueError(f"{pdb_id} not in cluster test set")

    full_set["redundency"] = full_set["job_name"].apply(get_redundency)
    print(f"len(full_set): {len(full_set)}")
    print(
        f"Test Independant Set (N): {len(clusters_test['test_independent_set'])}\n"
        f"Test Redundent Set (N): {len(clusters_test['test_redundent_set'])}\n"
        f"Train Redundent Set (N): {len(clusters_test['train_redundent_set'])}\n"
    )

    return full_set[full_set["redundency"] == 0].copy()

