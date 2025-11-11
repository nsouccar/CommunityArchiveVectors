"""
Download topic summary JSON files from Modal volume to local filesystem.
"""

import modal
import json
from pathlib import Path

# Modal setup
app = modal.App("download-summaries")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume})
def download_summary(year: int):
    """Download summary JSON for a specific year from Modal volume."""

    summary_path = Path(f"/data/topics_year_{year}_summary.json")

    if not summary_path.exists():
        print(f"❌ No summary found for year {year}")
        return None

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    print(f"✓ Found {year}: {len(summary['communities'])} communities")
    return summary


@app.local_entrypoint()
def main(years: str = "2012,2018,2019,2020,2021,2022,2023,2024,2025",
         output_dir: str = "frontend/public/data"):
    """Download summaries for multiple years."""

    year_list = [int(y.strip()) for y in years.split(',')]

    print(f"\n{'='*80}")
    print(f"DOWNLOADING TOPIC SUMMARIES")
    print(f"{'='*80}")
    print(f"Years: {year_list}")
    print(f"Output: {output_dir}")

    all_summaries = {}

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for year in year_list:
        result = download_summary.remote(year)
        if result:
            all_summaries[year] = result

            # Save individual year file
            year_file = output_path / f"topics_year_{year}_summary.json"
            with open(year_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Saved {year_file}")

    # Create combined summary
    combined_path = output_path / "all_topics.json"
    with open(combined_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")
    print(f"Downloaded {len(all_summaries)} years")
    print(f"Combined file: {combined_path}")
