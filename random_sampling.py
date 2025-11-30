import os
import random
import time
import pandas as pd
import io
import json
import zstandard as zstd
import logging

# '../RawData/_raw_yt_metadata.jsonl.zst' has 85 million rows. Reduce the sample size for testing.
RATIO = 0.01  # e.g., 0.1 means 10% sample
PATH_TO_FOLDER = "../RawData"
FILE_NAME = "_raw_yt_metadata.jsonl.zst"
SRC_PATH = os.path.join(PATH_TO_FOLDER, FILE_NAME)
OUTPUT_BASE = "../SampleData/raw_yt_metadata_random_sample"
MAX_LINES = None  # e.g., set to an int to cap lines read in sampling pass
LOG_EVERY = 1_000_000  # progress logging cadence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

start_time = time.time()
logging.info("Starting random sampling")
logging.info(f"Source file: {SRC_PATH}")
logging.info(f"Sampling ratio: {RATIO}")

# 1) Count total rows (first pass)
total_rows = 0
dctx = zstd.ZstdDecompressor()
logging.info(f"Counting total rows in source file")
with open(SRC_PATH, 'rb') as f:
    with dctx.stream_reader(f) as reader:
        buffer = b''
        while True:
            chunk = reader.read(65536)
            if not chunk:
                break
            total_rows += chunk.count(b'\n')
            if total_rows % 1000000 == 0:
                print(f"Counted {total_rows:,} lines...")

print(f"Total rows in source: {total_rows}")
logging.info(f"Counted total rows: {total_rows}")

# Compute target samples for an early stop
target_samples = max(1, int(total_rows * RATIO))
logging.info(f"Target samples (approx): {target_samples}")

# 2) Sample rows in a second pass using Bernoulli sampling with stop condition
selected = []
lines_read = 0
malformed_count = 0

with open(SRC_PATH, 'rb') as fh:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(fh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        for line in text_stream:
            lines_read += 1

            # Optional MAX_LINES cap
            if MAX_LINES is not None and lines_read >= MAX_LINES:
                logging.warning(f"Reached MAX_LINES cap: {MAX_LINES}. Stopping early.")
                break

            if random.random() < RATIO:
                try:
                    selected.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed_count += 1
                    continue  # skip malformed lines

            # Progress logging
            if lines_read % LOG_EVERY == 0:
                logging.info(f"Read {lines_read} lines | Collected {len(selected)} samples | Malformed {malformed_count}")

            # Early stop if we have enough samples
            if len(selected) >= target_samples:
                logging.info(f"Collected target samples: {len(selected)}. Stopping early.")
                break

sample_df = pd.DataFrame(selected)

# 3) Save results as CSV and GZIP
gz_path  = f"{OUTPUT_BASE}.csv.gz"
sample_df.to_csv(gz_path, index=False, compression='gzip')

duration = time.time() - start_time

# Prints
print(f"Sampled rows (~{RATIO*100:.1f}% target): {len(sample_df)}")
print(f"Malformed lines skipped: {malformed_count}")
print(f"Saved: {gz_path}")
print(f"Elapsed time: {duration:.2f} seconds")

# Logs
logging.info(f"Final samples collected: {len(sample_df)}")
logging.info(f"Malformed lines skipped: {malformed_count}")
logging.info(f"Saved GZIP: {gz_path}")
logging.info(f"Elapsed time: {duration:.2f} seconds")
logging.info("Random sampling completed")