"""Quick script to inspect what data is in the embedding batches"""
import modal
import pickle

app = modal.App("inspect-batch")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume})
def inspect_first_batch():
    """Look at the first batch file to see what data we have"""

    batch_path = "/data/batches/batch_0.pkl"

    print(f"Loading {batch_path}...")
    with open(batch_path, 'rb') as f:
        batch_data = pickle.load(f)

    print(f"\n=== BATCH FILE STRUCTURE ===")
    print(f"Keys in batch: {list(batch_data.keys())}")
    print()

    if 'metadata' in batch_data:
        print(f"Number of items in metadata: {len(batch_data['metadata'])}")
        print()
        print(f"=== SAMPLE METADATA (first 3 items) ===")
        for i, meta in enumerate(batch_data['metadata'][:3]):
            print(f"\nItem {i}:")
            print(f"  Keys: {list(meta.keys())}")
            for key, value in meta.items():
                if key == 'text':
                    print(f"  {key}: {str(value)[:80]}...")
                else:
                    print(f"  {key}: {value}")

    if 'embeddings' in batch_data:
        print(f"\n=== EMBEDDING INFO ===")
        print(f"Number of embeddings: {len(batch_data['embeddings'])}")
        print(f"Embedding shape: {batch_data['embeddings'][0].shape}")
        print(f"Embedding dtype: {batch_data['embeddings'][0].dtype}")

    print()
    print("=" * 60)
    print("SUMMARY:")
    print(f"  Metadata fields: {list(batch_data['metadata'][0].keys())}")
    print(f"  Embedding dimensions: {batch_data['embeddings'][0].shape[0]}")
    print("=" * 60)

    return {
        'metadata_fields': list(batch_data['metadata'][0].keys()),
        'embedding_dims': int(batch_data['embeddings'][0].shape[0]),
        'num_items': len(batch_data['metadata'])
    }

@app.local_entrypoint()
def main():
    result = inspect_first_batch.remote()
    print(f"\nResult: {result}")
