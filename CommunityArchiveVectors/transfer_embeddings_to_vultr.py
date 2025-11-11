"""
Transfer embeddings from Modal volume directly to Vultr server
"""
import modal

app = modal.App("transfer-to-vultr")

volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

image = modal.Image.debian_slim().apt_install("rsync", "openssh-client")

@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,  # 2 hours
    cpu=4,
)
def transfer_embeddings_to_vultr(vultr_host: str, vultr_path: str, ssh_key: str = None):
    """
    Compress and transfer embeddings to Vultr server

    Args:
        vultr_host: Vultr server address (e.g. "root@45.63.18.97")
        vultr_path: Destination path on Vultr (e.g. "/root/embeddings")
        ssh_key: Optional SSH private key content
    """
    import subprocess
    import os
    import time

    print("="*80)
    print("STEP 1: COMPRESS BATCHES DIRECTORY")
    print("="*80)

    # Check batches directory
    batches_dir = "/data/batches"
    if not os.path.exists(batches_dir):
        return {"error": "Batches directory not found"}

    # Count files and size
    files = os.listdir(batches_dir)
    print(f"\nFound {len(files)} batch files")

    total_size = 0
    for f in files:
        path = os.path.join(batches_dir, f)
        if os.path.isfile(path):
            total_size += os.path.getsize(path)

    total_gb = total_size / (1024**3)
    print(f"Total size: {total_gb:.1f} GB")

    # Create compressed archive
    print("\nCompressing batches directory...")
    print("This may take 10-15 minutes...")

    start_time = time.time()

    tar_file = "/tmp/embeddings_batches.tar.gz"
    cmd = f"tar -czf {tar_file} -C /data batches/"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        return {"error": f"Compression failed: {result.stderr}"}

    compress_time = time.time() - start_time
    compressed_size = os.path.getsize(tar_file) / (1024**3)

    print(f"\n✓ Compressed in {compress_time:.1f} seconds")
    print(f"  Original: {total_gb:.1f} GB")
    print(f"  Compressed: {compressed_size:.1f} GB")
    print(f"  Compression ratio: {(1 - compressed_size/total_gb)*100:.1f}%")

    print("\n" + "="*80)
    print("STEP 2: TRANSFER TO VULTR")
    print("="*80)

    # Setup SSH key if provided
    if ssh_key:
        ssh_key_path = "/tmp/vultr_key"
        with open(ssh_key_path, 'w') as f:
            f.write(ssh_key)
        os.chmod(ssh_key_path, 0o600)
        ssh_opts = f"-i {ssh_key_path}"
    else:
        ssh_opts = ""

    # Transfer using rsync with progress
    print(f"\nTransferring to {vultr_host}:{vultr_path}")
    print("This may take 10-30 minutes depending on network speed...")

    # Create destination directory on Vultr
    mkdir_cmd = f"ssh {ssh_opts} {vultr_host} 'mkdir -p {vultr_path}'"
    subprocess.run(mkdir_cmd, shell=True)

    # Transfer file
    start_time = time.time()

    rsync_cmd = f"rsync -avz --progress -e 'ssh {ssh_opts}' {tar_file} {vultr_host}:{vultr_path}/"

    result = subprocess.run(rsync_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        return {"error": f"Transfer failed: {result.stderr}"}

    transfer_time = time.time() - start_time
    transfer_speed_mbps = (compressed_size * 1024 * 8) / transfer_time

    print(f"\n✓ Transfer complete in {transfer_time:.1f} seconds")
    print(f"  Speed: {transfer_speed_mbps:.1f} Mbps")

    print("\n" + "="*80)
    print("STEP 3: EXTRACT ON VULTR")
    print("="*80)

    # Extract on Vultr
    print("\nExtracting archive on Vultr...")

    extract_cmd = f"ssh {ssh_opts} {vultr_host} 'cd {vultr_path} && tar -xzf embeddings_batches.tar.gz && rm embeddings_batches.tar.gz'"

    result = subprocess.run(extract_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        return {"error": f"Extraction failed: {result.stderr}"}

    print("✓ Extraction complete")

    # Verify
    verify_cmd = f"ssh {ssh_opts} {vultr_host} 'ls -lh {vultr_path}/batches | head -20'"
    result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)

    print("\nFiles on Vultr:")
    print(result.stdout)

    total_time = compress_time + transfer_time

    return {
        "success": True,
        "original_size_gb": total_gb,
        "compressed_size_gb": compressed_size,
        "compression_time_sec": compress_time,
        "transfer_time_sec": transfer_time,
        "total_time_sec": total_time,
        "transfer_speed_mbps": transfer_speed_mbps,
        "destination": f"{vultr_host}:{vultr_path}/batches"
    }

@app.local_entrypoint()
def main(vultr_host: str = "root@45.63.18.97", vultr_path: str = "/root/tweet-search/embeddings"):
    """
    Transfer embeddings to Vultr

    Args:
        vultr_host: Vultr server address
        vultr_path: Destination path on Vultr
    """
    print("="*80)
    print("TRANSFERRING EMBEDDINGS TO VULTR")
    print("="*80)
    print(f"\nSource: Modal volume 'tweet-vectors-large'")
    print(f"Destination: {vultr_host}:{vultr_path}")
    print("\nThis will:")
    print("  1. Compress 58GB of batch files (~10-15 min)")
    print("  2. Transfer to Vultr (~10-30 min)")
    print("  3. Extract on Vultr (~5 min)")
    print("\nTotal time: ~25-50 minutes")

    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled")
        return

    result = transfer_embeddings_to_vultr.remote(vultr_host, vultr_path)

    if "error" in result:
        print(f"\n❌ ERROR: {result['error']}")
        return

    print("\n" + "="*80)
    print("✅ TRANSFER COMPLETE!")
    print("="*80)
    print(f"\nOriginal size: {result['original_size_gb']:.1f} GB")
    print(f"Compressed size: {result['compressed_size_gb']:.1f} GB")
    print(f"Compression time: {result['compression_time_sec']/60:.1f} minutes")
    print(f"Transfer time: {result['transfer_time_sec']/60:.1f} minutes")
    print(f"Total time: {result['total_time_sec']/60:.1f} minutes")
    print(f"Transfer speed: {result['transfer_speed_mbps']:.1f} Mbps")
    print(f"\nEmbeddings are now at: {result['destination']}")
    print("\nNext: Run clustering script on Vultr server!")
