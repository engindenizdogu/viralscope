# trendy-tube
Predictive exploration of YouTube channel & video performance.

### To Do
- [IN PROGRESS] Create a sample dataset
- [IN PROGRESS] Create the preprocessing pipeline
   - One hot encoding
   - Scaling
   - Feature engineering/extraction/selection
- Model training
   - Random Forest Model
   - Find 1 or 2 more models to try
- Model Evaluation
- (Optional) Look for additional scariping solutions (for more up-to-date data)
- Agentic Architecture
- Frontend

### Data Assets (repo root / `RawData/`)
| File | Status |
|------|--------|
| `_raw_df_channels.tsv.gz` | sampled |
| `_raw_df_timeseries.tsv.gz` | sampled |
| `_raw_yt_metadata.jsonl.zst` | downloading |
| `df_channels_en.tsv.gz` | sampled |
| `df_timeseries_en.tsv.gz` | sampled |
| `num_comments.tsv.gz` | sampled |
| `num_comments_authors.tsv.gz` | downloading |
| `youtube_comments.tsv.gz` | downloading |
| `yt_metadata_en.jsonl.gz` | downloading |
| `yt_metadata_helper.feather` | downloading |

### Sampling Large Files
```python
import pandas as pd

FILE_NAME = 'num_comments'
PATH = f'RawData/{FILE_NAME}.tsv.gz'

SAMPLE_SIZE = 50_000
CHUNK_SIZE = 1_000_000  # read subset to avoid full load

# Read chunk, then sample
df = pd.read_csv(PATH, sep='\t', compression='gzip', nrows=CHUNK_SIZE)
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# Save
df_sample.to_csv(f'SampleData/{FILE_NAME}_sample.csv', index=False)
```

### Next Steps
- Feature engineering & model training (RandomForest + others)
- Pipeline: encoding, scaling, selection
- Experiment tracking & evaluation
