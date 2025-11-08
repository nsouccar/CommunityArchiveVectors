#!/usr/bin/env python3
"""
Download CoreNN database from Modal Volume to local storage

This backs up your 33GB database before we migrate to Hetzner.
"""
import modal
import os

app = modal.App("download-database")

image = modal.Image.debian_slim()
vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=3600,  # 1 hour timeout for large download
)
def download_to_local():
    """
    Download the database from Modal Volume

    This will create a tarball that you can download locally
    """
    import tarfile
    import time

    print("=" * 80)
    print("📦 Creating backup of CoreNN database and metadata")
    print("=" * 80)

    # Reload volume to get latest
    vector_volume.reload()

    # Check what files exist
    print("\n📂 Checking files in volume...")
    import subprocess
    result = subprocess.run(["ls", "-lah", "/data"], capture_output=True, text=True)
    print(result.stdout)

    # Create tarball
    print("\n🗜️  Creating tarball (this will take a while for 33GB)...")
    start = time.time()

    with tarfile.open("/tmp/corenn_backup.tar.gz", "w:gz") as tar:
        # Add database
        if os.path.exists("/data/corenn_db"):
            print("   Adding corenn_db/...")
            tar.add("/data/corenn_db", arcname="corenn_db")

        # Add metadata
        if os.path.exists("/data/metadata.pkl"):
            print("   Adding metadata.pkl...")
            tar.add("/data/metadata.pkl", arcname="metadata.pkl")

    duration = time.time() - start

    # Get size
    size_bytes = os.path.getsize("/tmp/corenn_backup.tar.gz")
    size_gb = size_bytes / (1024**3)

    print(f"\n✅ Backup created in {duration:.1f}s")
    print(f"   Size: {size_gb:.2f} GB")
    print(f"   Location: /tmp/corenn_backup.tar.gz")

    print("\n" + "=" * 80)
    print("💡 To download this file:")
    print("   modal volume get tweet-vectors-volume /corenn_backup.tar.gz ./corenn_backup.tar.gz")
    print("=" * 80)

    # Copy to volume so we can download it
    print("\n📤 Copying tarball to volume for download...")
    import shutil
    shutil.copy2("/tmp/corenn_backup.tar.gz", "/data/corenn_backup.tar.gz")

    print("\n✅ Backup complete and ready to download!")

    return {
        "size_gb": size_gb,
        "duration_seconds": duration,
        "files": ["corenn_db/", "metadata.pkl"]
    }


@app.local_entrypoint()
def main():
    """Create backup and instructions for download"""
    print("🚀 Starting database backup process...\n")

    result = download_to_local.remote()

    print("\n" + "=" * 80)
    print("✅ BACKUP COMPLETE!")
    print("=" * 80)
    print(f"\nBackup details:")
    print(f"  - Size: {result['size_gb']:.2f} GB")
    print(f"  - Time: {result['duration_seconds']:.1f} seconds")
    print(f"  - Files: {', '.join(result['files'])}")

    print("\n📥 To download the backup to your computer, run:")
    print("   modal volume get tweet-vectors-volume /corenn_backup.tar.gz ./corenn_backup.tar.gz")

    print("\n📤 Once downloaded, extract with:")
    print("   tar -xzf corenn_backup.tar.gz")

    print("\n💾 This gives you a local backup before migrating to Hetzner!")
    print("=" * 80)
