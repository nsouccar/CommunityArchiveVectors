"""
Direct transfer of embeddings to Vultr (no compression)
Fast cloud-to-cloud transfer using rsync
"""

import modal
import subprocess

# Create Modal app and volume
app = modal.App("transfer-embeddings-direct")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

# Image with rsync and SSH
image = modal.Image.debian_slim().run_commands(
    "apt-get update",
    "apt-get install -y rsync openssh-client"
)

@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=3600,  # 1 hour should be plenty
    cpu=2,
)
def direct_transfer_embeddings(vultr_host: str, vultr_path: str):
    """
    Transfer embeddings directly without compression
    Much faster for data center to data center transfers
    """
    import os
    import time

    print("=" * 80)
    print("DIRECT TRANSFER TO VULTR (NO COMPRESSION)")
    print("=" * 80)
    print()
    print(f"Source: Modal volume 'tweet-vectors-large'")
    print(f"Destination: {vultr_host}:{vultr_path}")
    print()

    # Check batches directory
    batches_dir = "/data/batches"
    if not os.path.exists(batches_dir):
        raise ValueError(f"Batches directory not found: {batches_dir}")

    batch_files = sorted([f for f in os.listdir(batches_dir) if f.endswith('.pkl')])
    total_size_gb = sum(os.path.getsize(os.path.join(batches_dir, f)) for f in batch_files) / (1024**3)

    print(f"Found {len(batch_files)} batch files")
    print(f"Total size: {total_size_gb:.1f} GB")
    print()
    print("Transfer method: Direct rsync (no compression)")
    print("Estimated time: 10-20 minutes at 100 MB/s")
    print()

    # SSH options
    ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

    # Create destination directory
    print("=" * 80)
    print("STEP 1: CREATE DESTINATION DIRECTORY")
    print("=" * 80)
    print()

    mkdir_cmd = f"ssh {ssh_opts} {vultr_host} 'mkdir -p {vultr_path}/batches'"
    print(f"Running: {mkdir_cmd}")
    result = subprocess.run(mkdir_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError("Failed to create destination directory")
    print("✓ Directory created")
    print()

    # Transfer with rsync (direct, no compression)
    print("=" * 80)
    print("STEP 2: TRANSFER EMBEDDINGS")
    print("=" * 80)
    print()
    print(f"Transferring {total_size_gb:.1f} GB...")
    print()

    start_time = time.time()

    # Use rsync with progress
    # -a: archive mode (preserves permissions, timestamps)
    # -v: verbose
    # --progress: show progress
    # --partial: keep partially transferred files
    rsync_cmd = f"rsync -av --progress --partial -e 'ssh {ssh_opts}' {batches_dir}/ {vultr_host}:{vultr_path}/batches/"

    print(f"Running: rsync -av --progress --partial")
    print()

    process = subprocess.Popen(
        rsync_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Print output in real-time
    for line in process.stdout:
        print(line, end='')

    process.wait()

    if process.returncode != 0:
        raise RuntimeError("rsync failed")

    elapsed = time.time() - start_time
    speed_mbps = (total_size_gb * 1024) / elapsed

    print()
    print("=" * 80)
    print("TRANSFER COMPLETE!")
    print("=" * 80)
    print()
    print(f"Time taken: {elapsed / 60:.1f} minutes")
    print(f"Average speed: {speed_mbps:.1f} MB/s")
    print(f"Transferred: {total_size_gb:.1f} GB")
    print()
    print(f"Files are now at: {vultr_host}:{vultr_path}/batches/")
    print()

@app.local_entrypoint()
def main():
    """Transfer embeddings to Vultr server"""
    vultr_host = "root@45.63.18.97"
    vultr_path = "/root/tweet-search/embeddings"

    print()
    print("=" * 80)
    print("DIRECT TRANSFER (NO COMPRESSION)")
    print("=" * 80)
    print()
    print(f"This will transfer 56.5 GB directly to Vultr")
    print(f"Estimated time: 10-20 minutes")
    print()

    confirm = input("Proceed? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Transfer cancelled")
        return

    direct_transfer_embeddings.remote(vultr_host, vultr_path)

    print()
    print("=" * 80)
    print("ALL DONE!")
    print("=" * 80)
