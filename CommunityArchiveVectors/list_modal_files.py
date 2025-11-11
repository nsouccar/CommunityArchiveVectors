"""
List all files in Modal volumes to find raw embeddings
"""
import modal

app = modal.App("list-volume-files")

# Check all three volumes
volumes = [
    "tweet-vectors-large",
    "tweet-vectors-volume",
    "tweet-vectors-storage"
]

@app.function(timeout=600)
def list_files_in_volume(volume_name: str):
    """List files in a Modal volume"""
    import os

    vol = modal.Volume.from_name(volume_name, create_if_missing=False)

    # Mount the volume
    with vol.batch_upload() as batch:
        pass  # Just to ensure it's accessible

    results = {
        'volume': volume_name,
        'files': [],
        'dirs': []
    }

    # List files
    for entry in vol.listdir("/"):
        results['files'].append(entry.path)
        if entry.type == "directory":
            results['dirs'].append(entry.path)

    return results

@app.local_entrypoint()
def main():
    """Check all volumes for raw embeddings"""
    print("="*80)
    print("CHECKING MODAL VOLUMES FOR RAW EMBEDDINGS")
    print("="*80)

    for vol_name in volumes:
        print(f"\n{'='*80}")
        print(f"Volume: {vol_name}")
        print("="*80)

        try:
            result = list_files_in_volume.remote(vol_name)

            print(f"\nFiles and directories found:")
            for item in result['files']:
                print(f"  - {item}")

            # Look for embedding files
            embedding_files = [f for f in result['files'] if any(ext in f for ext in ['.npy', '.pkl', '.bin', 'embed'])]
            if embedding_files:
                print(f"\n✓ Found {len(embedding_files)} potential embedding files!")
                for f in embedding_files[:10]:
                    print(f"  → {f}")
        except Exception as e:
            print(f"  Error: {e}")
