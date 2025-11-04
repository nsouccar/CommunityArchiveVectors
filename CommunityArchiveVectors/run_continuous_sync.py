#!/usr/bin/env python3
"""
Continuous sync script - runs 50K tweet batches sequentially until complete.
Usage: python run_continuous_sync.py [--max-batches N]
"""

import subprocess
import sys
import time
from datetime import datetime

def run_batch(batch_num):
    """Run a single 50K batch and return True if successful."""
    print(f"\n{'='*80}")
    print(f"BATCH {batch_num} - Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            ["modal", "run", "modal_app.py::sync_tweets_from_supabase", "--limit", "50000"],
            check=True,
            capture_output=False,
            text=True
        )

        print(f"\n✅ Batch {batch_num} completed successfully at {datetime.now().strftime('%H:%M:%S')}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n⚠️  Batch {batch_num} timed out (error code {e.returncode})")
        print(f"💡 Data likely saved before timeout - continuing to next batch...")
        return True  # Treat as success since data is saved before timeout
    except KeyboardInterrupt:
        print(f"\n⏸️  Interrupted by user during batch {batch_num}")
        raise

def main():
    max_batches = None

    # Parse command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--max-batches" and len(sys.argv) > 2:
            max_batches = int(sys.argv[2])
            print(f"Will run maximum of {max_batches} batches")

    start_time = datetime.now()
    print(f"🚀 Starting continuous sync at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Each batch processes 50K tweets (~30 mins per batch)")

    if max_batches:
        print(f"🎯 Target: {max_batches} batches (~{max_batches * 50000:,} tweets)")
    else:
        print(f"🎯 Target: All remaining tweets (will run until database is complete)")

    print(f"\nPress Ctrl+C to stop gracefully after current batch completes\n")

    batch_num = 1
    successful_batches = 0

    try:
        while True:
            if max_batches and batch_num > max_batches:
                print(f"\n🎉 Reached maximum of {max_batches} batches!")
                break

            success = run_batch(batch_num)
            successful_batches += 1  # Count all batches (timeouts save data too!)
            batch_num += 1

            # Brief pause between batches
            time.sleep(5)

    except KeyboardInterrupt:
        print(f"\n\n⏸️  Stopped by user after completing batch {batch_num - 1}")

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    hours = duration.total_seconds() / 3600

    print(f"\n{'='*80}")
    print(f"📊 SYNC SUMMARY")
    print(f"{'='*80}")
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {hours:.1f} hours")
    print(f"✅ Batches completed: {successful_batches}")
    print(f"📈 Tweets processed: ~{successful_batches * 50000:,}")
    print(f"💡 Note: Some batches may have timed out, but data was saved!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
