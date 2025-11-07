#!/usr/bin/env python3
"""
Quick script to check exactly which batch the offline builder is on
"""

import modal

app = modal.App("check-progress")
vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=True)


@app.function(volumes={"/data": vector_volume}, timeout=60)
def check_progress():
    """Check how many batches have been completed"""
    import glob
    import pickle
    import os

    vector_volume.reload()

    if not os.path.exists("/data/batches"):
        return 0, 0

    batch_files = sorted(glob.glob("/data/batches/batch_*.pkl"))

    if not batch_files:
        return 0, 0

    total_embeddings = 0
    for batch_file in batch_files:
        with open(batch_file, "rb") as f:
            data = pickle.load(f)
            total_embeddings += data["count"]

    return len(batch_files), total_embeddings


@app.local_entrypoint()
def main():
    num_batches, total_embeddings = check_progress.remote()

    print(f"\n{'='*60}")
    print(f"OFFLINE BUILDER PROGRESS")
    print(f"{'='*60}")
    print(f"Batches completed: {num_batches}")
    print(f"Total embeddings: {total_embeddings:,}")
    print(f"Progress: ~{(total_embeddings / 7_000_000) * 100:.1f}% of estimated 7M")
    print(f"{'='*60}\n")
