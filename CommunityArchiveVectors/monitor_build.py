#!/usr/bin/env python3
"""
Real-time monitor for CoreNN incremental build progress.
Parses logs and displays visual progress tracking.
"""

import re
import sys
from datetime import datetime, timedelta

def parse_batch_logs(log_text):
    """Extract batch timing data from logs."""
    batches = []

    # Pattern to match: "✅ Inserted in 157.8s (634 vectors/sec)"
    pattern = r'Batch (\d+)/64:.*?✅ Inserted in ([\d.]+)s \((\d+) vectors/sec\)'

    matches = re.findall(pattern, log_text, re.DOTALL)

    for batch_num, time_sec, vecs_per_sec in matches:
        batches.append({
            'batch': int(batch_num),
            'time': float(time_sec),
            'vps': int(vecs_per_sec),
            'vectors': int(batch_num) * 100000
        })

    return batches

def create_ascii_chart(batches, width=60):
    """Create ASCII bar chart of batch timings."""
    if not batches:
        return ""

    max_time = max(b['time'] for b in batches)

    lines = []
    lines.append("\n" + "="*70)
    lines.append("BATCH TIMING CHART (seconds per 100K vectors)")
    lines.append("="*70)

    for b in batches:
        bar_len = int((b['time'] / max_time) * width)
        bar = "█" * bar_len
        line = f"Batch {b['batch']:2d}: {bar} {b['time']:.1f}s ({b['vps']} v/s)"
        lines.append(line)

    return "\n".join(lines)

def calculate_projection(batches):
    """Calculate time projection based on trends."""
    if len(batches) < 3:
        return None

    # Recent batches (last 5)
    recent = batches[-5:] if len(batches) >= 5 else batches
    avg_recent = sum(b['time'] for b in recent) / len(recent)

    # Check if increasing
    first_half_avg = sum(b['time'] for b in batches[:len(batches)//2]) / (len(batches)//2)
    second_half_avg = sum(b['time'] for b in batches[len(batches)//2:]) / (len(batches) - len(batches)//2)

    is_increasing = second_half_avg > first_half_avg
    increase_rate = (second_half_avg - first_half_avg) / first_half_avg if is_increasing else 0

    remaining_batches = 64 - len(batches)

    # Conservative projection assuming continued slowdown
    if is_increasing and increase_rate > 0.1:
        # Assume linear increase
        projected_avg = avg_recent * (1 + increase_rate * 0.5)
    else:
        projected_avg = avg_recent

    remaining_seconds = remaining_batches * projected_avg

    return {
        'remaining_batches': remaining_batches,
        'avg_recent': avg_recent,
        'projected_avg': projected_avg,
        'remaining_time': remaining_seconds,
        'is_increasing': is_increasing,
        'increase_rate': increase_rate
    }

def format_time(seconds):
    """Format seconds into human readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def display_progress(batches):
    """Display comprehensive progress report."""
    if not batches:
        print("\n❌ No batch data found yet. Build may still be initializing.\n")
        return

    latest = batches[-1]
    total_completed = len(batches)
    progress_pct = (total_completed / 64) * 100

    print("\n" + "="*70)
    print("CORENN INCREMENTAL BUILD PROGRESS")
    print("="*70)

    # Progress bar
    bar_width = 50
    filled = int(bar_width * total_completed / 64)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n{bar} {progress_pct:.1f}%")
    print(f"Completed: {total_completed}/64 batches ({latest['vectors']:,} vectors)\n")

    # Current status
    print(f"Latest Batch: {latest['batch']}")
    print(f"  Time: {latest['time']:.1f}s ({latest['vps']} vectors/sec)")

    # Timing trend
    if len(batches) >= 2:
        prev = batches[-2]
        change = ((latest['time'] - prev['time']) / prev['time']) * 100
        trend = "📈" if change > 5 else "📉" if change < -5 else "➡️"
        print(f"  vs Previous: {change:+.1f}% {trend}")

    # Chart
    print(create_ascii_chart(batches))

    # Projection
    proj = calculate_projection(batches)
    if proj:
        print("\n" + "="*70)
        print("TIME PROJECTION")
        print("="*70)
        print(f"Remaining batches: {proj['remaining_batches']}")
        print(f"Recent average: {proj['avg_recent']:.1f}s per batch")

        if proj['is_increasing']:
            print(f"⚠️  Trend: INCREASING (+{proj['increase_rate']*100:.1f}%)")
            print(f"Projected average: {proj['projected_avg']:.1f}s per batch")
        else:
            print(f"✅ Trend: STABLE")

        print(f"\nEstimated time remaining: {format_time(proj['remaining_time'])}")

        eta = datetime.now() + timedelta(seconds=proj['remaining_time'])
        print(f"Estimated completion: {eta.strftime('%I:%M %p')}")

        # Warning if slow
        if proj['projected_avg'] > 600:
            print("\n⚠️  WARNING: Batches averaging > 10 min. Build may take 12+ hours.")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    # Read from stdin or file
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            log_text = f.read()
    else:
        log_text = sys.stdin.read()

    batches = parse_batch_logs(log_text)
    display_progress(batches)
