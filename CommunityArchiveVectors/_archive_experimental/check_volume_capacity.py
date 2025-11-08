#!/usr/bin/env python3
"""
Check the actual capacity and available space on the Modal Volume.
"""

import modal

app = modal.App("check-volume-capacity")

image = modal.Image.debian_slim()

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=300,
)
def check_volume_capacity():
    import subprocess
    import os

    vector_volume.reload()

    print("="*80)
    print("MODAL VOLUME CAPACITY CHECK")
    print("="*80 + "\n")

    # Check disk space on the volume mount
    result = subprocess.run(["df", "-h", "/data"], capture_output=True, text=True)
    print("Volume mounted at /data:")
    print(result.stdout)
    print()

    # Parse the output
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        header = lines[0]
        data = lines[1]
        parts = data.split()

        filesystem = parts[0]
        total = parts[1]
        used = parts[2]
        available = parts[3]
        use_pct = parts[4]

        print(f"Filesystem: {filesystem}")
        print(f"Total Capacity: {total}")
        print(f"Currently Used: {used}")
        print(f"Available Space: {available}")
        print(f"Usage Percentage: {use_pct}\n")

        # Convert to numbers for calculation
        def parse_size(s):
            """Convert size string like '33G' to GB number"""
            if s.endswith('G'):
                return float(s[:-1])
            elif s.endswith('M'):
                return float(s[:-1]) / 1024
            elif s.endswith('K'):
                return float(s[:-1]) / 1024 / 1024
            elif s.endswith('T'):
                return float(s[:-1]) * 1024
            return float(s) / 1024 / 1024 / 1024

        total_gb = parse_size(total)
        used_gb = parse_size(used)
        available_gb = parse_size(available)

        print("="*80)
        print("ANALYSIS")
        print("="*80 + "\n")

        print(f"Current situation:")
        print(f"  - Database: ~33GB")
        print(f"  - Batch files: ~57GB")
        print(f"  - Total used: {used_gb:.1f}GB\n")

        print(f"After completing build:")
        print(f"  - Final database: ~43GB (estimate)")
        print(f"  - Batch files: ~57GB")
        print(f"  - Total needed: ~100GB\n")

        if available_gb < 15:
            print(f"❌ NOT ENOUGH SPACE!")
            print(f"   Available: {available_gb:.1f}GB")
            print(f"   Needed for completion: ~15GB more")
            print(f"   Modal Volume capacity: {total_gb:.1f}GB")
            print(f"\n💡 Solution: Volume capacity is the bottleneck")
            print(f"   Options:")
            print(f"   1. Contact Modal support to expand volume")
            print(f"   2. Use ephemeral-only strategy (keep DB there, lose on shutdown)")
            print(f"   3. Switch to managed vector DB (Pinecone, Qdrant)")
        else:
            print(f"✅ ENOUGH SPACE AVAILABLE!")
            print(f"   Available: {available_gb:.1f}GB")
            print(f"   Needed for completion: ~15GB")
            print(f"   Safe to proceed with ephemeral strategy!")

    print("\n" + "="*80 + "\n")


@app.local_entrypoint()
def main():
    check_volume_capacity.remote()
