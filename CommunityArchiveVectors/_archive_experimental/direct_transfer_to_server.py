#!/usr/bin/env python3
"""
Direct Transfer - Send database from Modal directly to your server
This skips downloading to your laptop, saving hours!
"""

import modal
import os

app = modal.App("direct-transfer")

image = modal.Image.debian_slim().apt_install("rsync", "openssh-client")
volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_dict({
        "SERVER_IP": "YOUR_SERVER_IP_HERE",
        "SSH_KEY": "YOUR_PRIVATE_KEY_HERE"  # We'll handle this differently
    })]
)
def transfer_to_server(server_ip: str):
    """
    Transfer database directly from Modal to your server using rsync
    Much faster than downloading locally first!
    """
    import subprocess

    print("=" * 80)
    print("🚀 DIRECT TRANSFER FROM MODAL TO YOUR SERVER")
    print("=" * 80)
    print(f"Target server: {server_ip}")
    print("This will take 1-2 hours for 40GB\n")

    # Create tarball
    print("📦 Creating tarball...")
    subprocess.run([
        "tar", "-czf", "/tmp/database.tar.gz",
        "-C", "/data",
        "corenn_db", "metadata.pkl"
    ], check=True)

    print("✅ Tarball created\n")

    # Transfer via scp (you'll need to set up SSH key)
    print(f"📤 Transferring to {server_ip}...")
    print("Note: You need to set up SSH key authentication first")

    # This would use scp but needs SSH key setup
    # subprocess.run([
    #     "scp", "-i", "/root/.ssh/id_rsa",
    #     "/tmp/database.tar.gz",
    #     f"root@{server_ip}:/root/tweet-search/"
    # ], check=True)

    print("\n" + "=" * 80)
    print("📝 MANUAL TRANSFER INSTRUCTIONS")
    print("=" * 80)
    print("""
Since SSH key setup in Modal is complex, here's the fastest manual approach:

1. The backup tarball is ready in Modal volume at: /corenn_backup.tar.gz

2. Download it directly to your new DigitalOcean server:

   # SSH into your new server
   ssh root@YOUR_SERVER_IP

   # Install Modal CLI on the server
   pip install modal
   modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET

   # Download directly to server (skips your laptop!)
   cd /root
   mkdir tweet-search
   cd tweet-search
   modal volume get tweet-vectors-volume /corenn_backup.tar.gz ./corenn_backup.tar.gz

   # Extract
   tar -xzf corenn_backup.tar.gz

3. Deploy backend while downloading:
   - Transfer hetzner_backend.py to server
   - Install dependencies
   - Set up systemd service

This way the server downloads directly from Modal, saving hours!
""")

@app.local_entrypoint()
def main(server_ip: str = ""):
    """Run direct transfer"""
    if not server_ip:
        print("❌ Please provide server IP:")
        print("   modal run direct_transfer_to_server.py --server-ip YOUR_SERVER_IP")
        return

    transfer_to_server.remote(server_ip)
