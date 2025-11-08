#!/usr/bin/env python3
"""
Copy database files to root of volume for easy download
"""

import modal

app = modal.App("copy-database")
image = modal.Image.debian_slim()
volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=7200,  # 2 hours
)
def copy_database_to_root():
    """Copy corenn_db and metadata.pkl to /download/ for easy access"""
    import subprocess
    import os

    print("=" * 80)
    print("COPYING DATABASE FILES FOR DOWNLOAD")
    print("=" * 80)

    # Create download directory at volume root
    os.makedirs("/data/download", exist_ok=True)

    # Copy metadata.pkl (small, fast)
    print("\n📋 Copying metadata.pkl...")
    if os.path.exists("/data/metadata.pkl"):
        subprocess.run(["cp", "/data/metadata.pkl", "/data/download/"], check=True)
        print("✅ metadata.pkl copied")
    else:
        print("❌ metadata.pkl not found!")

    # Copy corenn_db directory (large, will take time)
    print("\n📦 Copying corenn_db/ directory (39GB - will take ~10 minutes)...")
    if os.path.exists("/data/corenn_db"):
        subprocess.run(["cp", "-r", "/data/corenn_db", "/data/download/"], check=True)
        print("✅ corenn_db/ copied")
    else:
        print("❌ corenn_db/ not found!")

    # Commit changes
    print("\n💾 Committing changes to volume...")
    volume.commit()

    # List what we have
    print("\n📂 Contents of /data/download/:")
    subprocess.run(["ls", "-lh", "/data/download/"])

    print("\n" + "=" * 80)
    print("✅ DATABASE READY FOR DOWNLOAD!")
    print("=" * 80)
    print("\nYou can now download:")
    print("  modal volume get tweet-vectors-volume /download/metadata.pkl")
    print("  modal volume get tweet-vectors-volume /download/corenn_db")
    print("=" * 80)

    return True

@app.local_entrypoint()
def main():
    """Run the copy"""
    print("\nStarting database file copy...")
    result = copy_database_to_root.remote()
    if result:
        print("\n🎉 Success! Database files ready for download.")
    else:
        print("\n❌ Copy failed!")
