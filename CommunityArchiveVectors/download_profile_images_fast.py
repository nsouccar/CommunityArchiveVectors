"""
Fast profile image downloader with progress updates.
Downloads images in parallel batches for speed.
"""

import json
import os
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set
import time

def get_unique_usernames() -> Set[str]:
    """Extract unique usernames from network animation data."""
    network_file = Path("frontend/public/network_animation_data.json")

    print(f"Loading {network_file}...")
    with open(network_file, 'r') as f:
        data = json.load(f)

    usernames = set()
    for year_data in data.get("years", []):
        for node in year_data.get("nodes", []):
            username = node.get("id")
            if username:
                usernames.add(username)

    print(f"Found {len(usernames)} unique users")
    return usernames

def download_image(username: str, output_dir: Path) -> tuple[str, bool, str]:
    """Download a single profile image. Returns (username, success, message)."""
    output_path = output_dir / f"{username}.jpg"

    # Skip if already downloaded
    if output_path.exists():
        return (username, True, "skipped")

    url = f"https://unavatar.io/x/{username}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        return (username, True, "downloaded")
    except Exception as e:
        return (username, False, str(e))

def download_images_parallel(usernames: Set[str], output_dir: Path, max_workers: int = 20):
    """Download images in parallel with progress updates."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(usernames)
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"\nDownloading {total} images using {max_workers} parallel workers...")
    print("Progress updates every 100 images:")
    print()

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_username = {
            executor.submit(download_image, username, output_dir): username
            for username in usernames
        }

        # Process as they complete
        for i, future in enumerate(as_completed(future_to_username), 1):
            username, success, message = future.result()

            if message == "skipped":
                skipped += 1
            elif success:
                downloaded += 1
            else:
                failed += 1

            # Progress update every 100 images
            if i % 100 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0

                print(f"  {i}/{total} processed ({downloaded} new, {skipped} exist, {failed} failed) "
                      f"- {rate:.1f} images/sec - ETA: {remaining/60:.1f} min")

    elapsed = time.time() - start_time

    print(f"\n✓ Download complete in {elapsed:.1f} seconds!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Already existed: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total ready: {downloaded + skipped} images")

def main():
    print("="*80)
    print("DOWNLOADING PROFILE IMAGES FOR NETWORK VISUALIZATION")
    print("="*80 + "\n")

    # Step 1: Extract unique usernames
    usernames = get_unique_usernames()

    # Step 2: Download images in parallel
    output_dir = Path("frontend/public/profile-images")
    download_images_parallel(usernames, output_dir)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
