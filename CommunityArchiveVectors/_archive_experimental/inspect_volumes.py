#!/usr/bin/env python3
"""
Inspect all Modal volumes to see what's in each one
"""
import modal

app = modal.App("inspect-volumes")
image = modal.Image.debian_slim()

# Volume 1: tweet-vectors-large
volume_large = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

# Volume 2: tweet-vectors-volume
volume_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

# Volume 3: tweet-vectors-storage
volume_storage = modal.Volume.from_name("tweet-vectors-storage", create_if_missing=False)


@app.function(
    image=image,
    volumes={
        "/large": volume_large,
        "/volume": volume_volume,
        "/storage": volume_storage,
    },
    timeout=300,
)
def inspect_all_volumes():
    """Check contents of all volumes"""
    import subprocess
    import os

    print("=" * 80)
    print("INSPECTING ALL MODAL VOLUMES")
    print("=" * 80)

    volumes = {
        "tweet-vectors-large": "/large",
        "tweet-vectors-volume": "/volume",
        "tweet-vectors-storage": "/storage",
    }

    for name, path in volumes.items():
        print(f"\n{'=' * 80}")
        print(f"📦 VOLUME: {name}")
        print(f"{'=' * 80}")

        if not os.path.exists(path):
            print(f"❌ Path {path} does not exist")
            continue

        # List contents
        print(f"\n📂 Contents of {path}:")
        result = subprocess.run(["ls", "-lah", path], capture_output=True, text=True)
        print(result.stdout)

        # Get total size
        result = subprocess.run(["du", "-sh", path], capture_output=True, text=True)
        size = result.stdout.strip().split()[0] if result.stdout else "unknown"
        print(f"\n💾 Total size: {size}")

        # Check for specific files
        files_to_check = [
            "corenn_db",
            "metadata.pkl",
            "corenn_backup.tar.gz",
        ]

        print(f"\n🔍 Checking for important files:")
        for file in files_to_check:
            full_path = os.path.join(path, file)
            if os.path.exists(full_path):
                if os.path.isdir(full_path):
                    result = subprocess.run(["du", "-sh", full_path], capture_output=True, text=True)
                    size = result.stdout.strip().split()[0] if result.stdout else "unknown"
                    print(f"  ✅ {file}/ (directory, size: {size})")
                else:
                    size_bytes = os.path.getsize(full_path)
                    size_gb = size_bytes / (1024**3)
                    print(f"  ✅ {file} (file, size: {size_gb:.2f} GB)")
            else:
                print(f"  ❌ {file} (not found)")

    print(f"\n{'=' * 80}")
    print("INSPECTION COMPLETE")
    print(f"{'=' * 80}\n")


@app.local_entrypoint()
def main():
    """Run volume inspection"""
    inspect_all_volumes.remote()
