"""
Combine locally filtered topic summary files into all_topics.json
WITHOUT downloading from Modal (which would overwrite filtered data)
"""

import json
from pathlib import Path

def main():
    years = [2012, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    data_dir = Path("frontend/public/data")

    all_summaries = {}

    print(f"\n{'='*80}")
    print(f"COMBINING LOCAL FILTERED SUMMARIES")
    print(f"{'='*80}")

    for year in years:
        year_file = data_dir / f"topics_year_{year}_summary.json"

        if year_file.exists():
            with open(year_file, 'r') as f:
                summary = json.load(f)
            all_summaries[year] = summary
            print(f"✓ Loaded {year}: {len(summary['communities'])} communities")
        else:
            print(f"⚠ Skipped {year}: file not found")

    # Save combined file
    combined_path = data_dir / "all_topics.json"
    with open(combined_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")
    print(f"Combined {len(all_summaries)} years into {combined_path}")

if __name__ == "__main__":
    main()
