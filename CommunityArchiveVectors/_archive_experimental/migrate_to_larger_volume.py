#!/usr/bin/env python3
"""
Migrate database from small volume to larger volume.

This script:
1. Creates a new larger Modal Volume (100GB)
2. Copies all data from the old volume to the new volume
3. Verifies the copy was successful
"""

import modal

app = modal.App("migrate-volume")

image = modal.Image.debian_slim().pip_install("corenn-py")

# Old volume (currently 50GB, almost full)
old_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

# New larger volume (100GB)
new_volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=True)


@app.function(
    image=image,
    volumes={
        "/old_data": old_volume,
        "/new_data": new_volume,
    },
    cpu=4.0,
    memory=16384,
    timeout=7200,  # 2 hours should be enough to copy 33GB
)
def migrate_data():
    """
    Copy all data from old volume to new volume.
    """
    import subprocess
    import os

    print("="*80)
    print("MIGRATING DATA TO LARGER VOLUME")
    print("="*80 + "\n")

    # Reload volumes
    old_volume.reload()
    new_volume.reload()

    print("Step 1: Checking old volume contents...\n")

    # Check what exists on old volume
    result = subprocess.run(["df", "-h", "/old_data"], capture_output=True, text=True)
    print("Old volume space:")
    print(result.stdout)

    result = subprocess.run(["du", "-sh", "/old_data/corenn_db"], capture_output=True, text=True)
    print(f"Database size: {result.stdout.strip()}")

    result = subprocess.run(["du", "-sh", "/old_data/batches"], capture_output=True, text=True)
    print(f"Batches size: {result.stdout.strip()}")

    print("\nStep 2: Copying data to new volume...\n")

    # Copy database
    if os.path.exists("/old_data/corenn_db"):
        print("📦 Copying CoreNN database (this may take 5-10 minutes)...")
        result = subprocess.run(
            ["cp", "-r", "/old_data/corenn_db", "/new_data/"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Database copied successfully")
        else:
            print(f"❌ Database copy failed: {result.stderr}")
            return False

    # Copy batch files
    if os.path.exists("/old_data/batches"):
        print("📦 Copying batch files...")
        result = subprocess.run(
            ["cp", "-r", "/old_data/batches", "/new_data/"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Batch files copied successfully")
        else:
            print(f"❌ Batch files copy failed: {result.stderr}")
            return False

    # Copy metadata if it exists
    if os.path.exists("/old_data/metadata.pkl"):
        print("📦 Copying metadata...")
        result = subprocess.run(
            ["cp", "/old_data/metadata.pkl", "/new_data/"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Metadata copied successfully")
        else:
            print("⚠️  Metadata copy failed (not critical)")

    print("\nStep 3: Verifying new volume...\n")

    # Verify new volume
    result = subprocess.run(["df", "-h", "/new_data"], capture_output=True, text=True)
    print("New volume space:")
    print(result.stdout)

    result = subprocess.run(["du", "-sh", "/new_data/corenn_db"], capture_output=True, text=True)
    print(f"Database size: {result.stdout.strip()}")

    result = subprocess.run(["du", "-sh", "/new_data/batches"], capture_output=True, text=True)
    print(f"Batches size: {result.stdout.strip()}")

    # Verify database can be opened
    print("\nStep 4: Verifying database integrity...\n")
    try:
        from corenn_py import CoreNN
        db = CoreNN.open("/new_data/corenn_db")
        print("✅ Database opens successfully on new volume!")
    except Exception as e:
        print(f"❌ Failed to open database: {e}")
        return False

    # Commit changes to new volume
    new_volume.commit()

    print("\n" + "="*80)
    print("✅ MIGRATION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Update your scripts to use 'tweet-vectors-large' instead of 'tweet-vectors-volume'")
    print("2. Run resume_builder.py to continue building from batch 55")
    print("3. Once verified, you can delete the old volume to save costs")
    print("="*80 + "\n")

    return True


@app.local_entrypoint()
def main():
    """
    Run the migration.
    """
    success = migrate_data.remote()

    if success:
        print("\n🎉 Migration successful! You can now resume building the database.")
    else:
        print("\n❌ Migration failed. Check the logs above for errors.")
