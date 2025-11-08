#!/usr/bin/env python3
"""
Smart self-healing continuous sync script.
Automatically detects retry loops and reduces chunk size to fix issues.
"""

import subprocess
import sys
import time
import re
from datetime import datetime
from collections import defaultdict

def update_chunk_size(new_size):
    """Update CHUNK_SIZE in modal_app.py"""
    print(f"\n🔧 Reducing chunk size to {new_size} vectors...")

    with open("modal_app.py", "r") as f:
        content = f.read()

    # Find and replace CHUNK_SIZE
    pattern = r'CHUNK_SIZE = \d+  # Add \d+ vectors'
    replacement = f'CHUNK_SIZE = {new_size}  # Add {new_size} vectors'
    new_content = re.sub(pattern, replacement, content)

    with open("modal_app.py", "w") as f:
        f.write(new_content)

    print(f"✅ Updated modal_app.py: CHUNK_SIZE = {new_size}")

def run_batch_with_monitoring(batch_num, chunk_size):
    """Run a single batch and monitor for retry loops"""
    print(f"\n{'='*80}")
    print(f"BATCH {batch_num} - Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current chunk size: {chunk_size} vectors")
    print(f"{'='*80}\n")

    # Start the modal process
    process = subprocess.Popen(
        ["modal", "run", "modal_app.py::sync_tweets_from_supabase", "--limit", "10000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    current_tweet_id = None
    timeout_detected = False

    # Monitor output in real-time
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line, end='')  # Print output in real-time

            # Extract tweet_id from "Starting from tweet_id > X" line
            match = re.search(r'Starting from tweet_id > (\d+)', line)
            if match:
                current_tweet_id = match.group(1)

            # Detect timeout
            if 'Runner heartbeat timeout' in line or 'Runner failed' in line:
                timeout_detected = True

    process.wait()

    return {
        'success': process.returncode == 0,
        'timeout': timeout_detected,
        'tweet_id': current_tweet_id
    }

def main():
    start_time = datetime.now()
    print(f"🚀 Starting smart self-healing sync at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Each batch processes 10K tweets")
    print(f"🔧 Auto-adjusts chunk size when retry loops detected\n")

    batch_num = 1
    successful_batches = 0
    chunk_size = 1000  # Start with 1000

    # Track retry loops: tweet_id -> number of consecutive failures
    retry_tracker = defaultdict(int)
    last_tweet_id = None

    try:
        while True:
            result = run_batch_with_monitoring(batch_num, chunk_size)

            # Check for retry loop
            if result['timeout'] and result['tweet_id']:
                if result['tweet_id'] == last_tweet_id:
                    retry_tracker[result['tweet_id']] += 1

                    # Detect retry loop (3 consecutive failures on same tweet_id)
                    if retry_tracker[result['tweet_id']] >= 3:
                        print(f"\n⚠️  RETRY LOOP DETECTED on tweet_id {result['tweet_id']}")
                        print(f"Failed {retry_tracker[result['tweet_id']]} times on this batch")

                        # Auto-adjust chunk size
                        if chunk_size >= 1000:
                            chunk_size = 500
                        elif chunk_size >= 500:
                            chunk_size = 250
                        elif chunk_size >= 250:
                            chunk_size = 100
                        else:
                            print(f"❌ Chunk size already at minimum (100). Cannot reduce further.")
                            print(f"Manual intervention may be required.")
                            time.sleep(60)  # Wait a minute before retrying
                            continue

                        update_chunk_size(chunk_size)
                        retry_tracker.clear()  # Clear retry tracker after fix
                        print(f"🔄 Retrying with smaller chunks...\n")
                        time.sleep(5)
                        continue
                else:
                    # Different tweet_id - clear retry tracker
                    retry_tracker.clear()

                last_tweet_id = result['tweet_id']

            elif result['success']:
                # Success - clear retry tracker and continue
                successful_batches += 1
                retry_tracker.clear()
                last_tweet_id = None
                print(f"\n✅ Batch {batch_num} completed successfully!")

            batch_num += 1
            time.sleep(3)  # Brief pause between batches

    except KeyboardInterrupt:
        print(f"\n\n⏸️  Stopped by user after batch {batch_num - 1}")

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
    print(f"✅ Successful batches: {successful_batches}")
    print(f"📈 Tweets processed: ~{successful_batches * 10000:,}")
    print(f"🔧 Final chunk size: {chunk_size} vectors")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
