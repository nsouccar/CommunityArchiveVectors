#!/usr/bin/env python3
"""
Check if the raw embedding batch files still exist in the volume
"""
import modal

app = modal.App("check-batch-files")
image = modal.Image.debian_slim()
volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def check_batch_directory():
    """Check if /data/batches/ still exists with the raw embedding files"""
    import subprocess
    import os

    print("=" * 80)
    print("CHECKING FOR RAW EMBEDDING BATCH FILES")
    print("=" * 80)

    # Check if /data/batches exists
    if not os.path.exists("/data/batches"):
        print("\n❌ /data/batches/ directory DOES NOT exist")
        print("   The raw embedding batch files were likely deleted after database build")
        print("\n✅ This is NORMAL - batch files are temporary and deleted to save space")
        return

    print("\n✅ /data/batches/ directory EXISTS!\n")

    # List batch files
    result = subprocess.run(["ls", "-lh", "/data/batches"], capture_output=True, text=True)
    print("📂 Contents of /data/batches/:")
    print(result.stdout)

    # Count batch files
    result = subprocess.run(["find", "/data/batches", "-name", "batch_*.pkl"],
                          capture_output=True, text=True)
    batch_files = result.stdout.strip().split('\n')
    batch_count = len([f for f in batch_files if f])

    print(f"\n📊 Found {batch_count} batch files")

    # Get total size
    result = subprocess.run(["du", "-sh", "/data/batches"], capture_output=True, text=True)
    size = result.stdout.strip().split()[0] if result.stdout else "unknown"
    print(f"💾 Total size: {size}")

    print("\n" + "=" * 80)
    print("EXPLANATION:")
    print("=" * 80)
    print("""
These batch files contain the RAW embeddings that were used to BUILD the database:
- Each .pkl file has: embeddings (list of 1024-dim vectors) + metadata
- These were generated in 100K tweet batches
- ALL batches were loaded into RAM and used to build corenn_db/ in one shot

If these files still exist, they're taking up extra space (same data as corenn_db/)
You could DELETE them to save space, but keep them if you want a backup of raw embeddings.
""")
    print("=" * 80)

@app.local_entrypoint()
def main():
    """Run check"""
    check_batch_directory.remote()
