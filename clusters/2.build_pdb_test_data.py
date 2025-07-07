import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

# Load the CSV file
df = pd.read_csv("../job_list.csv")

# Create the output directory
pdb_data_dir = Path("./pdb_test_data")
pdb_data_dir.mkdir(exist_ok=True)

# Define the source mmCIF directory
mmcif_dir = Path("../pepdb")

# Define a function to get the cif file path by pdb_id
def get_cif_by_job_name(job_name):
    pdb_id = job_name.split("_")[0]
    cif_path = mmcif_dir / job_name / f"{pdb_id}.cif"
    return cif_path

# Initialize counters and error tracking
success_num = 0
error_num = 0
error_list = []

# Process each pdb_id
for pdb_id in tqdm(df.job_name):
    try:
        # Source and destination paths
        source_path = get_cif_by_job_name(pdb_id)
        destination_path = pdb_data_dir / source_path.name
        
        # Check if the file already exists
        if destination_path.exists():
            success_num += 1  # Count as success if file exists
            continue
        
        # Copy file if it doesn't exist
        shutil.copy(source_path, destination_path)
        success_num += 1
    except Exception as e:
        print(f"{pdb_id} failed to copy. Error: {e}")
        error_num += 1
        error_list.append(pdb_id)

# Print results
print(f"Success count: {success_num}")
print(f"Error count: {error_num}")
if error_list:
    print("Errors occurred for the following pdb_ids:")
    print(error_list)