"""
Download community names JSON files from Modal volume to local filesystem.
"""

import modal
import json
from pathlib import Path

# Modal setup
app = modal.App("download-community-names")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume})
def download_names(year: int):
    """Download community names JSON for a specific year from Modal volume."""

    names_path = Path(f"/data/community_names_{year}.json")

    if not names_path.exists():
        print(f"❌ No community names found for year {year}")
        return None

    with open(names_path, 'r') as f:
        names = json.load(f)

    print(f"✓ Found {year}: {len(names['communities'])} communities")
    return names


@app.local_entrypoint()
def main(years: str = "2012,2018,2019,2020,2021,2022,2023,2024,2025",
         output_dir: str = "frontend/public/data"):
    """Download community names for multiple years."""

    year_list = [int(y.strip()) for y in years.split(',')]

    print(f"\n{'='*80}")
    print(f"DOWNLOADING COMMUNITY NAMES")
    print(f"{'='*80}")
    print(f"Years: {year_list}")
    print(f"Output: {output_dir}")

    all_names = {}

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for year in year_list:
        result = download_names.remote(year)
        if result:
            all_names[year] = result

            # Save individual year file
            year_file = output_path / f"community_names_{year}.json"
            with open(year_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Saved {year_file}")

    # Create combined file
    combined_path = output_path / "all_community_names.json"
    with open(combined_path, 'w') as f:
        json.dump(all_names, f, indent=2)

    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")
    print(f"Downloaded {len(all_names)} years")
    print(f"Combined file: {combined_path}")
