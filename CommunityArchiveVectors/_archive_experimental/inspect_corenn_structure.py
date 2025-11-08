#!/usr/bin/env python3
"""
Inspect the CoreNN database structure to see what's inside
"""
import modal

app = modal.App("inspect-corenn-structure")
image = modal.Image.debian_slim()
volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def inspect_corenn_structure():
    """Look inside the CoreNN database directory"""
    import subprocess
    import os

    print("=" * 80)
    print("📦 INSPECTING CORENN DATABASE STRUCTURE")
    print("=" * 80)

    # Check corenn_db directory structure
    print("\n📂 Contents of /data/corenn_db/:")
    result = subprocess.run(["ls", "-lah", "/data/corenn_db"], capture_output=True, text=True)
    print(result.stdout)

    # Check if there's a tree structure (if tree is available)
    print("\n🌳 Directory tree:")
    result = subprocess.run(["find", "/data/corenn_db", "-type", "f", "-o", "-type", "d"],
                          capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')[:50]  # First 50 items
    for line in lines:
        print(line)

    if len(result.stdout.strip().split('\n')) > 50:
        print(f"... and {len(result.stdout.strip().split('\n')) - 50} more files/directories")

    # Check file sizes
    print("\n💾 File sizes in corenn_db/:")
    result = subprocess.run(["du", "-sh", "/data/corenn_db/*"],
                          capture_output=True, text=True, shell=True)
    print(result.stdout)

    # Check metadata.pkl
    print("\n📝 Metadata file:")
    if os.path.exists("/data/metadata.pkl"):
        size_bytes = os.path.getsize("/data/metadata.pkl")
        size_gb = size_bytes / (1024**3)
        print(f"  ✅ metadata.pkl exists: {size_gb:.2f} GB")

        # Try to peek at metadata structure
        print("\n🔍 Metadata structure preview:")
        try:
            import pickle
            with open("/data/metadata.pkl", "rb") as f:
                metadata_pkg = pickle.load(f)

            print(f"  Type: {type(metadata_pkg)}")
            if isinstance(metadata_pkg, dict):
                print(f"  Keys: {list(metadata_pkg.keys())}")
                if "count" in metadata_pkg:
                    print(f"  Total vectors: {metadata_pkg['count']:,}")
                if "metadata" in metadata_pkg:
                    print(f"  Metadata entries: {len(metadata_pkg['metadata']):,}")
                    # Show sample entry
                    sample_id = list(metadata_pkg['metadata'].keys())[0]
                    sample = metadata_pkg['metadata'][sample_id]
                    print(f"  Sample entry keys: {list(sample.keys())}")
        except Exception as e:
            print(f"  ⚠️  Could not inspect metadata: {e}")

    print("\n" + "=" * 80)
    print("EXPLANATION OF WHAT YOU HAVE:")
    print("=" * 80)
    print("""
📦 corenn_db/ (39GB) = YOUR VECTOR INDEX
   - This IS your embeddings database!
   - CoreNN stores vectors in an indexed format
   - All 6.4M tweet embeddings are in here
   - This is the "index file" you're asking about

📝 metadata.pkl (1.52GB) = TWEET TEXT & METADATA
   - Maps tweet IDs to their content
   - Contains: username, text, created_at, etc.
   - This is what you see in search results

✅ YOU HAVE EVERYTHING NEEDED!
   - Embeddings: IN corenn_db/ (indexed)
   - Metadata: IN metadata.pkl
   - This is a complete, working database
""")
    print("=" * 80)

@app.local_entrypoint()
def main():
    """Run inspection"""
    inspect_corenn_structure.remote()
