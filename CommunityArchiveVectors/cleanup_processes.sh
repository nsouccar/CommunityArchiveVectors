#!/bin/bash
# Kill all old Modal/Bun/Node processes except the current incremental builder

echo "Cleaning up old background processes..."
echo ""

# The current incremental builder we want to KEEP
KEEP_INCREMENTAL="incremental_builder.py"

# Kill all old modal offline_builder processes
echo "Killing old offline_builder processes..."
pkill -f "modal run offline_builder.py"

# Kill all old modal_app sync processes
echo "Killing old modal_app sync processes..."
pkill -f "modal_app.py.*sync"
pkill -f "run_continuous_sync.py"
pkill -f "precompute_all_topics"

# Kill all old bun processes
echo "Killing old bun/node processes..."
pkill -f "bun generateCleanedEmbeddings"
pkill -f "bun addUsernamesToEmbeddings"
pkill -f "bun src/server/searchAPI"
pkill -f "node -e"

# Kill sleep processes
echo "Killing old sleep/wait processes..."
pkill -f "sleep 120"
pkill -f "sleep 180"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Current incremental_builder.py is still running."
echo "Run 'ps aux | grep modal | grep -v grep' to verify."
