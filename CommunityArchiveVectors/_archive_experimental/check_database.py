#!/usr/bin/env python3
"""
Check if the partial CoreNN database is still usable after the crash.
"""

import modal

app = modal.App("check-database")

image = (
    modal.Image.debian_slim()
    .pip_install("corenn-py", "numpy")
)

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=600,
)
def check_database_status():
    """
    Check if the CoreNN database exists and try to open it.
    """
    import os
    import glob
    from corenn_py import CoreNN

    vector_volume.reload()

    print("="*80)
    print("CHECKING DATABASE STATUS")
    print("="*80 + "\n")

    # Check what files exist
    print("📁 Files on volume:\n")

    if os.path.exists("/data/corenn_db"):
        print("✅ CoreNN database directory exists: /data/corenn_db")

        # List some files in the database
        db_files = glob.glob("/data/corenn_db/*")
        print(f"   Contains {len(db_files)} files")

        # Show size
        import subprocess
        result = subprocess.run(["du", "-sh", "/data/corenn_db"], capture_output=True, text=True)
        print(f"   Size: {result.stdout.strip()}")
    else:
        print("❌ CoreNN database directory not found")
        return {"exists": False, "usable": False}

    if os.path.exists("/data/metadata.pkl"):
        print("✅ Metadata file exists: /data/metadata.pkl")

        import pickle
        with open("/data/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            count = metadata.get("count", 0)
            print(f"   Metadata says: {count:,} vectors")
    else:
        print("⚠️  Metadata file not found")

    print("\n" + "="*80)
    print("ATTEMPTING TO OPEN DATABASE")
    print("="*80 + "\n")

    # Try to open the database
    try:
        print("🔓 Attempting to open database...")
        db = CoreNN.open("/data/corenn_db")
        print("✅ Database opened successfully!\n")

        # Try to get stats
        try:
            # Try a simple operation - search for a random vector
            import numpy as np
            test_vector = np.random.randn(1024).astype(np.float32)
            test_vector = test_vector / np.linalg.norm(test_vector)

            print("🔍 Testing search functionality...")
            results = db.search_f32(test_vector, k=10)
            print(f"✅ Search works! Found {len(results)} results\n")

            return {
                "exists": True,
                "usable": True,
                "can_search": True,
                "message": "Database is intact and fully functional!"
            }

        except Exception as e:
            print(f"⚠️  Search test failed: {e}\n")
            return {
                "exists": True,
                "usable": True,
                "can_search": False,
                "message": "Database exists but search may be limited"
            }

    except Exception as e:
        print(f"❌ Failed to open database: {e}\n")
        print("The database may be corrupted due to the crash.")
        return {
            "exists": True,
            "usable": False,
            "error": str(e)
        }


@app.local_entrypoint()
def main():
    """
    Check the database status.
    """
    result = check_database_status.remote()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if result.get("usable"):
        print("✅ Good news! The 5.4M vector database is still usable!")
        print("   You can use it for testing or continue building from here.")
    else:
        print("❌ Bad news: The database is corrupted and cannot be used.")
        print("   You'll need to start fresh with more storage space.")

    print("="*80 + "\n")
