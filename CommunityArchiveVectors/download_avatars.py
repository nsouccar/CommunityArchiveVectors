"""
Download profile avatars locally for fast loading.

This script downloads all profile images from Twitter and saves them
locally in frontend/public/avatars/, then updates avatar_urls.json
to point to local paths.

Usage:
    python download_avatars.py
"""

import json
import os
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlparse
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import ssl

# Configuration
AVATAR_URLS_FILE = Path("frontend/public/avatar_urls.json")
NETWORK_DATA_FILE = Path("frontend/public/network_animation_data.json")
OUTPUT_DIR = Path("frontend/public/avatars")
MAX_WORKERS = 20  # Concurrent downloads
TIMEOUT = 10  # seconds per request

# Create unverified SSL context for Twitter CDN
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def download_image(username: str, url: str, output_dir: Path) -> tuple:
    """Download a single image. Returns (username, success, local_path_or_error)."""
    try:
        # Determine file extension from URL
        parsed = urlparse(url)
        path_parts = parsed.path.split('.')
        ext = path_parts[-1] if len(path_parts) > 1 else 'jpg'
        if ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            ext = 'jpg'

        output_path = output_dir / f"{username}.{ext}"

        # Skip if already downloaded
        if output_path.exists():
            return username, True, f"/avatars/{username}.{ext}"

        # Download the image
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=TIMEOUT, context=ssl_context) as response:
            content = response.read()
            output_path.write_bytes(content)
            return username, True, f"/avatars/{username}.{ext}"

    except HTTPError as e:
        return username, False, f"HTTP {e.code}"
    except URLError as e:
        return username, False, str(e.reason)
    except Exception as e:
        return username, False, str(e)


def main():
    print("=" * 80)
    print("DOWNLOADING PROFILE AVATARS")
    print("=" * 80)

    # Load avatar URLs
    print("\nLoading avatar URLs...")
    with open(AVATAR_URLS_FILE) as f:
        avatar_urls = json.load(f)
    print(f"  Found {len(avatar_urls)} avatar URLs")

    # Load network data to get all usernames
    print("\nLoading network data...")
    with open(NETWORK_DATA_FILE) as f:
        network_data = json.load(f)

    # Collect all unique usernames
    all_usernames = set()
    for year in network_data.get("years", []):
        for node in year.get("nodes", []):
            if node.get("id"):
                all_usernames.add(node["id"])
    print(f"  Found {len(all_usernames)} unique users in network")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Prepare download list - only users with known avatar URLs
    to_download = []
    for username in all_usernames:
        if username in avatar_urls:
            url = avatar_urls[username]
            # Skip default profile images
            if "default_profile" not in url:
                to_download.append((username, url))

    print(f"\nWill download {len(to_download)} avatars (skipping default profiles)")

    # Track results
    success_count = 0
    fail_count = 0
    local_paths = {}

    start_time = time.time()

    # Download with thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_image, username, url, OUTPUT_DIR): username
            for username, url in to_download
        }

        for i, future in enumerate(as_completed(futures)):
            username, success, result = future.result()
            if success:
                success_count += 1
                local_paths[username] = result
            else:
                fail_count += 1

            # Progress update every 100 images
            if (i + 1) % 100 == 0 or (i + 1) == len(to_download):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Progress: {i + 1}/{len(to_download)} ({success_count} ok, {fail_count} failed) - {rate:.1f}/sec")

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("UPDATING avatar_urls.json")
    print("=" * 80)

    # Update avatar_urls.json with local paths
    updated_urls = {}
    for username in all_usernames:
        if username in local_paths:
            updated_urls[username] = local_paths[username]
        elif username in avatar_urls:
            # Keep remote URL as fallback
            updated_urls[username] = avatar_urls[username]

    # Save updated file
    with open(AVATAR_URLS_FILE, 'w') as f:
        json.dump(updated_urls, f, indent=2)

    print(f"\nUpdated {len(local_paths)} URLs to local paths")
    print(f"Kept {len(updated_urls) - len(local_paths)} remote URLs as fallback")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nDownloaded {success_count} avatars in {elapsed:.1f} seconds")
    print(f"Failed: {fail_count}")
    print(f"\nAvatars saved to: {OUTPUT_DIR}")
    print("avatar_urls.json updated with local paths")


if __name__ == "__main__":
    main()
