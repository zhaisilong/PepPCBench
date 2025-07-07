import argparse
import requests
import json
import os
import csv

# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fetch RCSB PDB entries based on a JSON query file.")
    parser.add_argument(
        "--query_file", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing the query parameters."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results.csv", 
        help="Output file name to save the results as CSV (default: 'results.csv')."
    )
    return parser.parse_args()

# API URL
url = "https://search.rcsb.org/rcsbsearch/v2/query"

# Function to load JSON query from a file
def load_query_from_file(query_file):
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"The query file '{query_file}' does not exist.")
    
    with open(query_file, "r") as file:
        try:
            query = json.load(file)
            return query
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in the file '{query_file}': {e}")

# Function to fetch paginated results
def fetch_all_results(api_url, base_query):
    all_results = []
    start = 0
    rows = base_query["request_options"]["paginate"]["rows"]
    
    while True:
        # Update pagination parameters
        base_query["request_options"]["paginate"]["start"] = start
        response = requests.post(api_url, json=base_query)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            break
        
        # Parse JSON response
        data = response.json()
        results = data.get("result_set", [])
        if not results:
            break  # Exit loop if no more results
        
        all_results.extend(results)
        print(f"Fetched {len(results)} results (start={start})")
        
        # Increment start index for next page
        start += rows
    
    return all_results

# Function to save results to CSV
def save_results_to_csv(results, output_file):
    # Extract the desired fields
    processed_results = [{"pdb_id": r["identifier"], "score": r["score"]} for r in results]

    # Write to CSV
    with open(output_file, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["pdb_id", "score"])
        writer.writeheader()
        writer.writerows(processed_results)

    print(f"Results saved to {output_file}")

def main(args):
    # Load query from JSON file
    try:
        query = load_query_from_file(args.query_file)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    # Fetch all results
    print("Fetching data...")
    all_data = fetch_all_results(url, query)

    # Save results to CSV
    save_results_to_csv(all_data, args.output)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)