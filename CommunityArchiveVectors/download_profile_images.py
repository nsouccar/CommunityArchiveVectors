"""
Download profile images for all users in the network visualization.

Extracts unique usernames from network_animation_data.json and downloads
their profile images from unavatar.io (same service used by the frontend).
"""

import json
import os
from pathlib import Path
import requests
from typing import Set
import time

def get_unique_usernames() -> Set[str]:
    """Extract unique usernames from network animation data."""
    network_file = Path("frontend/public/network_animation_data.json")

    with open(network_file, 'r') as f:
        data = json.load(f)

    usernames = set()

    # Extract from all years in the network data
    for year_data in data.get("years", []):
        for node in year_data.get("nodes", []):
            username = node.get("id")  # In network data, username is stored as "id"
            if username:
                usernames.add(username)

    print(f"Found {len(usernames)} unique users in network data")
    return usernames

def download_images(usernames: Set[str], output_dir: Path):
    """Download all profile images from unavatar.io."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(usernames)
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"\nDownloading {total} profile images from unavatar.io...")

    for username in sorted(usernames):
        output_path = output_dir / f"{username}.jpg"

        # Skip if already downloaded
        if output_path.exists():
            skipped += 1
            continue

        # Use unavatar.io (same as frontend)
        url = f"https://unavatar.io/x/{username}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                f.write(response.content)

            downloaded += 1

            # Progress update every 100 images
            if downloaded % 100 == 0:
                print(f"  Downloaded {downloaded}/{total} images...")

            # Small delay to avoid overwhelming unavatar.io
            time.sleep(0.05)  # 50ms delay = ~20 requests/second

        except Exception as e:
            print(f"  Failed to download {username}: {e}")
            failed += 1

    print(f"\n✓ Download complete!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {downloaded + skipped} images ready")

def main():
    print("="*80)
    print("DOWNLOADING PROFILE IMAGES FOR NETWORK VISUALIZATION")
    print("="*80 + "\n")

    # Step 1: Extract unique usernames
    usernames = get_unique_usernames()

    # Step 2: Download images from unavatar.io
    output_dir = Path("frontend/public/profile-images")
    download_images(usernames, output_dir)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
