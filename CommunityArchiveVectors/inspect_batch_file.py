"""
Inspect the structure of a batch file to understand how embeddings are stored
"""
import modal

app = modal.App("inspect-batch")

volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(
    volumes={"/data": volume},
    timeout=600,
)
def inspect_batch():
    """Load and inspect one batch file"""
    import pickle

    print("Loading batch_0001.pkl...")

    with open('/data/batches/batch_0001.pkl', 'rb') as f:
        batch_data = pickle.load(f)

    print(f"\nBatch data type: {type(batch_data)}")

    if isinstance(batch_data, dict):
        print(f"Number of entries: {len(batch_data)}")

        # Show first few keys
        keys = list(batch_data.keys())[:5]
        print(f"\nFirst 5 keys: {keys}")

        # Check structure of first entry
        if keys:
            first_key = keys[0]
            first_value = batch_data[first_key]

            print(f"\nFirst entry:")
            print(f"  Key: {first_key} (type: {type(first_key)})")
            print(f"  Value type: {type(first_value)}")

            if hasattr(first_value, 'shape'):
                print(f"  Value shape: {first_value.shape}")
            elif isinstance(first_value, (list, tuple)):
                print(f"  Value length: {len(first_value)}")

    elif isinstance(batch_data, (list, tuple)):
        print(f"Batch is a list/tuple with {len(batch_data)} items")
        if len(batch_data) > 0:
            print(f"First item type: {type(batch_data[0])}")

    return {
        'type': str(type(batch_data)),
        'num_entries': len(batch_data) if hasattr(batch_data, '__len__') else None,
        'sample_keys': list(batch_data.keys())[:10] if isinstance(batch_data, dict) else None
    }

@app.local_entrypoint()
def main():
    print("="*80)
    print("INSPECTING BATCH FILE")
    print("="*80)

    result = inspect_batch.remote()

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)
    print(f"Batch type: {result['type']}")
    if result['num_entries']:
        print(f"Number of entries: {result['num_entries']:,}")
    if result['sample_keys']:
        print(f"\nSample keys:")
        for key in result['sample_keys']:
            print(f"  - {key}")
