import json
import sys

def main(json_path: str):
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract the list of benchmark stats
    stats = data.get("benchmarks_stats")
    if stats is None:
        print("Error: no 'benchmarks_stats' key found in JSON.", file=sys.stderr)
        sys.exit(1)
    if not isinstance(stats, list) or len(stats) == 0:
        print("Error: 'benchmarks_stats' is empty or not a list.", file=sys.stderr)
        sys.exit(1)

    # Print the first benchmark stat, nicely formatted
    first_stat = stats[0]
    print(json.dumps(first_stat, indent=2))

if __name__ == "__main__":
    # Allow passing the JSON path on the command line; default to ./benchmarks.json
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks.json"
    main(path)
