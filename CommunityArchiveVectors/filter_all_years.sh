#!/bin/bash

# Filter tweets for all years sequentially
# This will take several hours to complete

source modal_env/bin/activate

YEARS=(2012 2019 2020 2021 2022 2023 2024 2025)

echo "======================================"
echo "Starting tweet filtering for all years"
echo "Years to process: ${YEARS[@]}"
echo "======================================"
echo ""

for year in "${YEARS[@]}"; do
    echo ""
    echo "======================================"
    echo "Processing year: $year"
    echo "======================================"
    echo ""

    modal run filter_cluster_tweets.py --year "$year"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully filtered year $year"
    else
        echo "✗ Error filtering year $year"
    fi

    echo ""
done

echo ""
echo "======================================"
echo "All years processed!"
echo "======================================"
