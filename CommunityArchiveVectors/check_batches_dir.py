"""
Check what's in the batches directory
"""
import modal

app = modal.App("check-batches")

volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(
    volumes={"/data": volume},
    timeout=600,
)
def check_batches():
    """Check what's in the batches directory"""
    import os

    print("Checking /data/batches directory...")

    if not os.path.exists('/data/batches'):
        return {"error": "batches directory does not exist"}

    # List files in batches
    files = os.listdir('/data/batches')

    results = {
        'num_files': len(files),
        'files': files[:20],  # Show first 20
        'file_sizes': {}
    }

    # Check file sizes
    for f in files[:10]:
        path = f'/data/batches/{f}'
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            results['file_sizes'][f] = f"{size_mb:.2f} MB"

    return results

@app.local_entrypoint()
def main():
    print("="*80)
    print("CHECKING BATCHES DIRECTORY")
    print("="*80)

    result = check_batches.remote()

    if 'error' in result:
        print(f"\nError: {result['error']}")
        return

    print(f"\nFound {result['num_files']} files in /data/batches/")
    print("\nFirst 20 files:")
    for f in result['files']:
        print(f"  - {f}")

    if result['file_sizes']:
        print("\nFile sizes (first 10):")
        for f, size in result['file_sizes'].items():
            print(f"  {f}: {size}")
