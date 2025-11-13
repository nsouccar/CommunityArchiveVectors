import modal
import json

app = modal.App("download-topics")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume})
def download_topics():
    """Download full topic summaries from Modal volume - including tweet_ids"""
    import os
    results = {}

    # Check what files exist
    data_dir = "/data"
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"Files in /data: {files}")

        # Download all summary JSON files WITH full topic data including tweet_ids
        for filename in files:
            if filename.startswith("topics_year_") and filename.endswith("_summary.json"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    year = filename.replace("topics_year_", "").replace("_summary.json", "")

                    # Return the FULL data structure with tweet_ids
                    results[year] = data

                    # Print stats
                    num_communities = len(data.get("communities", {}))
                    total_topics = sum(len(topics) for topics in data.get("communities", {}).values())
                    print(f"Loaded {filename} - {num_communities} communities, {total_topics} topics")
    else:
        print("Data directory doesn't exist yet")

    return results

@app.local_entrypoint()
def main():
    results = download_topics.remote()

    if results:
        # Save each year's data to individual files for the combine script
        for year, data in results.items():
            output_path = f'frontend/public/data/topics_year_{year}_summary.json'
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            num_communities = len(data.get('communities', {}))
            total_topics = sum(len(topics) for topics in data.get('communities', {}).values())
            print(f"\nYear {year}:")
            print(f"  Saved to: {output_path}")
            print(f"  Communities: {num_communities}")
            print(f"  Total topics: {total_topics}")

        print(f"\nDownloaded {len(results)} years")
        print(f"Files saved to frontend/public/data/")
        print(f"\nNext: Run python3 combine_local_topics.py to merge into all_topics.json")
    else:
        print("No topic files found yet")
