"""
Download all profile images from Supabase and create username mapping.
Gets avatar URLs from all_profile table and downloads them locally.
"""

import json
import os
from pathlib import Path
import requests
from supabase import create_client
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def get_username_avatar_mapping():
    """Get username to avatar URL mapping from Supabase."""
    print("Connecting to Supabase...")
    supabase = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY")
    )

    # First, check if username is in all_profile table
    print("Checking all_profile table structure...")
    sample = supabase.table("all_profile").select("*").limit(1).execute()

    if sample.data:
        print(f"all_profile fields: {list(sample.data[0].keys())}")

    # Get all profiles with avatars
    print("\nFetching profiles with avatars...")
    profiles = []
    page_size = 1000
    offset = 0

    while True:
        response = supabase.table("all_profile")\
            .select("account_id, avatar_media_url")\
            .not_("avatar_media_url", "is", None)\
            .range(offset, offset + page_size - 1)\
            .execute()

        if not response.data:
            break

        profiles.extend(response.data)
        print(f"  Fetched {len(profiles)} profiles...")

        if len(response.data) < page_size:
            break

        offset += page_size

    print(f"Total profiles with avatars: {len(profiles)}")

    # Now get usernames - need to join with tweets table
    print("\nFetching usernames from tweets table...")
    account_ids = [p['account_id'] for p in profiles]

    # Get unique username for each account_id
    username_map = {}
    batch_size = 100

    for i in range(0, len(account_ids), batch_size):
        batch = account_ids[i:i + batch_size]

        # Get one tweet per account to extract username
        response = supabase.table("tweets")\
            .select("account_id, account_display_name")\
            .in_("account_id", batch)\
            .execute()

        for tweet in response.data:
            account_id = tweet['account_id']
            username = tweet.get('account_display_name')
            if username and account_id not in username_map:
                username_map[account_id] = username

        print(f"  Mapped {len(username_map)} usernames...")

    # Combine profiles with usernames
    result = []
    for profile in profiles:
        account_id = profile['account_id']
        username = username_map.get(account_id)
        if username:
            result.append({
                'username': username,
                'avatar_url': profile['avatar_media_url']
            })

    print(f"\nSuccessfully mapped {len(result)} profiles to usernames")
    return result

def download_image(item: dict, output_dir: Path) -> tuple[str, bool, str]:
    """Download a single profile image. Returns (username, success, message)."""
    username = item['username']
    url = item['avatar_url']

    # Sanitize username for filesystem
    safe_username = username.replace('/', '_').replace('\\', '_')
    output_path = output_dir / f"{safe_username}.jpg"

    # Skip if already downloaded
    if output_path.exists():
        return (username, True, "skipped")

    try:
        # Get higher quality image (replace _normal with _400x400)
        url = url.replace('_normal', '_400x400')

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        return (username, True, "downloaded")
    except Exception as e:
        return (username, False, str(e))

def download_images_parallel(profiles: list, output_dir: Path, max_workers: int = 20):
    """Download images in parallel with progress updates."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(profiles)
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"\nDownloading {total} images using {max_workers} parallel workers...")
    print("Progress updates every 100 images:\n")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(download_image, item, output_dir): item
            for item in profiles
        }

        # Process as they complete
        for i, future in enumerate(as_completed(future_to_item), 1):
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

    return {
        'downloaded': downloaded,
        'skipped': skipped,
        'failed': failed
    }

def create_mapping_file(profiles: list, output_dir: Path):
    """Create a JSON mapping file from username to image filename."""
    mapping = {}

    for item in profiles:
        username = item['username']
        safe_username = username.replace('/', '_').replace('\\', '_')
        mapping[username] = f"{safe_username}.jpg"

    mapping_file = output_dir / "username_to_image.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\n✓ Created mapping file: {mapping_file}")
    print(f"  Maps {len(mapping)} usernames to image files")

def main():
    print("="*80)
    print("DOWNLOADING PROFILE IMAGES FROM SUPABASE")
    print("="*80 + "\n")

    # Step 1: Get username to avatar URL mapping from Supabase
    profiles = get_username_avatar_mapping()

    if not profiles:
        print("No profiles found!")
        return

    # Step 2: Download images in parallel
    output_dir = Path("frontend/public/profile-images")
    stats = download_images_parallel(profiles, output_dir)

    # Step 3: Create mapping file
    create_mapping_file(profiles, output_dir)

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nProfile images saved to: {output_dir}")
    print(f"Use username_to_image.json to map usernames to image files")

if __name__ == "__main__":
    main()
