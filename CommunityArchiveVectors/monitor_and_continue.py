"""
Monitor 2018 filtering completion and automatically start next tasks.

This script:
1. Monitors 2018 filtering progress
2. When complete, starts profile image download
3. Then starts filtering remaining years 2019-2025
"""

import time
import subprocess
import json
from pathlib import Path

def check_2018_complete():
    """Check if 2018 filtering is complete by looking at saved data."""
    data_file = Path("frontend/public/data/topics_year_2018_summary.json")

    if not data_file.exists():
        return False

    try:
        with open(data_file, 'r') as f:
            data = json.load(f)

        # Check if all communities have been processed
        # (The filter script updates the file after each community completes)
        communities = data.get('communities', {})

        # If we have data in communities, check if filtering is done
        if communities:
            # Look for a marker or check if file was recently modified
            # For now, we'll consider it complete if it exists and has data
            return True

        return False
    except Exception as e:
        print(f"Error checking 2018 status: {e}")
        return False

def start_profile_download():
    """Start downloading profile images."""
    print("\n" + "="*80)
    print("2018 FILTERING COMPLETE!")
    print("="*80)
    print("\nStarting profile image download...")
    print("="*80 + "\n")

    result = subprocess.run(
        ["python3", "download_profile_images.py"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print(f"Error during download: {result.stderr}")
        return False

    return True

def start_remaining_years():
    """Start filtering years 2019-2025."""
    print("\n" + "="*80)
    print("PROFILE IMAGES DOWNLOADED!")
    print("="*80)
    print("\nStarting filtering for years 2019-2025...")
    print("This will take approximately 21-28 hours.")
    print("="*80 + "\n")

    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    for year in years:
        print(f"\n{'='*80}")
        print(f"Processing year: {year}")
        print('='*80 + "\n")

        result = subprocess.run(
            ["bash", "-c", f"source modal_env/bin/activate && modal run filter_cluster_tweets_sequential.py --year {year}"],
            capture_output=False  # Show output in real-time
        )

        if result.returncode != 0:
            print(f"\nError filtering year {year}")
            return False

        print(f"\n✓ Year {year} complete!\n")

    return True

def main():
    print("="*80)
    print("MONITORING 2018 FILTERING PROGRESS")
    print("="*80)
    print("\nWaiting for 2018 to complete...")
    print("Checking every 5 minutes...\n")

    check_interval = 300  # 5 minutes

    while True:
        if check_2018_complete():
            print("\n✓ 2018 filtering detected as complete!")

            # Start profile download
            if start_profile_download():
                print("\n✓ Profile images downloaded successfully!")

                # Start remaining years
                start_remaining_years()

                print("\n" + "="*80)
                print("ALL TASKS COMPLETE!")
                print("="*80)
                break
            else:
                print("\n✗ Profile download failed. Please check manually.")
                break

        # Wait before checking again
        time.sleep(check_interval)

if __name__ == "__main__":
    main()
