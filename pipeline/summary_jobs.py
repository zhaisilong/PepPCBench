from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from peppcbench.utils import load_job_info
from peppcbench.multiprocess import get_status

if __name__ == "__main__":
    df = pd.read_csv("job_list.csv")
    data_dir = Path("pepdb")
    model_name_list = ["af3"]
    select_cols = ["job_name", "pdb_id", "peptide_chains", "protein_chains"]
    gen_fig = True

    all_infos = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        job_name = row["job_name"]
        job_dir = data_dir / job_name
        job_info = load_job_info(job_dir)

        peptide_chain_id = job_info.get("peptide_chains", None)
        peptide_chain_len = job_info.get(peptide_chain_id, {}).get("length", 0)
        peptide_chain_has_ncaa = job_info.get(peptide_chain_id, {}).get(
            "has_ncaa", False
        )

        filtered_info = {col: job_info.get(col, None) for col in select_cols}
        filtered_info.update(
            {
                "peptide_chain_len": peptide_chain_len,
                "peptide_chain_has_ncaa": peptide_chain_has_ncaa,
            }
        )
        if len(model_name_list) == 0:
            model_name_list = ["af3"]

        for model_name in model_name_list:
            status_dir = job_dir / model_name

            msa_status = get_status(status_dir, "msa")
            infer_status = get_status(status_dir, "infer")

            filtered_info[f"msa_status_{model_name}"] = msa_status
            filtered_info[f"infer_status_{model_name}"] = infer_status

            all_infos.append(filtered_info)

    df = pd.DataFrame(all_infos)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = results_dir / "job_summary.csv"
    df.to_csv(output_csv, index=False)

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
