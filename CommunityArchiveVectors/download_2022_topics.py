import modal
import json

app = modal.App("download-2022-topics")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume})
def download_2022_topics():
    """Download full topic data for 2022"""
    import os

    data_dir = "/data"

    # List all files first
    print("\n=== Files in /data directory ===")
    if os.path.exists(data_dir):
        all_files = sorted(os.listdir(data_dir))
        print(f"Total files: {len(all_files)}")
        year_2022_files = [f for f in all_files if '2022' in f or '2021' in f]
        print(f"\n2021/2022 files: {year_2022_files}")

    # Try different possible filenames
    possible_files = [
        "topics_year_2022_summary.json",
        "topics_2022_summary.json",
        "year_2022_topics.json",
        "2022_topics.json"
    ]

    for filename in possible_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"\nFound file: {filename}")
            with open(filepath, 'r') as f:
                data = json.load(f)
            print(f"Data keys: {list(data.keys())}")
            print(f"Communities: {data.get('total_communities', 0)}")
            print(f"Topics: {len(data.get('topics', []))}")
            return data

    print(f"\nNo 2022 topic files found")
    return None

@app.local_entrypoint()
def main():
    result = download_2022_topics.remote()

    if result:
        # Save full data to local file
        output_path = 'frontend/public/data/topics_2022.json'
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {output_path}")
        print(f"Total topics: {len(result.get('topics', []))}")
    else:
        print("No 2022 topic data found")
