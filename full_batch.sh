#!/bin/bash
set -e

# Go to project directory (important for relative paths)
cd "$(dirname "$0")"

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate image_classifier

# Logging
echo "======================" >> logs/cronlog.log
echo "Full batch run started at $(date)" >> logs/cronlog.log

# Start Flask API in the background
echo "Starting Flask-API..." >> logs/cronlog.log
python scripts/app.py >> logs/cronlog.log 2>&1 &
FLASK_PID=$!

# Short pause so that Flask can start
sleep 5

# Run batch processing
echo "Running batch processing..." >> logs/cronlog.log
python scripts/batch_processing.py >> logs/cronlog.log 2>&1

# Stop Flask
kill $FLASK_PID
echo "Flask-API stopped." >> logs/cronlog.log

# End log
echo "Full batch run finished at $(date)" >> logs/cronlog.log
echo "======================" >> logs/cronlog.log