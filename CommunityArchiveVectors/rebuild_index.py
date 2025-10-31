"""
Rebuild FAISS index with HNSW for faster search
This converts the existing flat index to HNSW index
"""

import modal
from modal_app import app, vector_volume, secrets, image

@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vector_volume},
    timeout=1800
)
def rebuild_with_hnsw():
    """Rebuild existing index with HNSW for faster search"""
    import faiss
    import pickle
    import os

    print("🔄 Rebuilding index with HNSW for faster search...")

    # Load existing flat index
    index_path = "/data/index.faiss"
    metadata_path = "/data/metadata.pkl"

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return {"error": "No existing index found"}

    print("📂 Loading existing index...")
    old_index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"✅ Loaded {old_index.ntotal:,} vectors")

    # Create new HNSW index
    print("🔨 Creating HNSW index...")
    dimension = old_index.d
    new_index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    new_index.hnsw.efConstruction = 200

    # Copy vectors from old index to new index
    print("📥 Transferring vectors to HNSW index...")
    vectors = old_index.reconstruct_n(0, old_index.ntotal)
    new_index.add(vectors)

    print(f"✅ Added {new_index.ntotal:,} vectors to HNSW index")

    # Save new index
    print("💾 Saving HNSW index...")
    faiss.write_index(new_index, index_path)

    # Metadata stays the same
    print("✅ Index rebuilt successfully!")

    vector_volume.commit()

    return {
        "status": "success",
        "vectors": new_index.ntotal,
        "index_type": "HNSW",
        "parameters": {
            "M": 32,
            "efConstruction": 200
        }
    }

@app.local_entrypoint()
def main():
    """Rebuild the index with HNSW"""
    print("Starting index rebuild...")
    result = rebuild_with_hnsw.remote()
    print(f"\nResult: {result}")
