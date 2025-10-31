"""Clear the vector index to start fresh"""
import modal
from modal_app import app, vector_volume, image

@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=60
)
def clear_index():
    """Delete the existing index files"""
    import os

    index_path = "/data/index.faiss"
    metadata_path = "/data/metadata.pkl"

    deleted = []
    if os.path.exists(index_path):
        os.remove(index_path)
        deleted.append("index.faiss")

    if os.path.exists(metadata_path):
        os.remove(metadata_path)
        deleted.append("metadata.pkl")

    vector_volume.commit()

    return {"status": "success", "deleted": deleted}

@app.local_entrypoint()
def main():
    result = clear_index.remote()
    print(f"Cleared: {result}")
