# trendy-tube
Predictive exploration of YouTube channel & video performance.

## Table of Contents
- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Python Scripts](#python-scripts)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Workflow](#pipeline-workflow)
- [Data Assets](#data-assets)
- [Project Roadmap](#project-roadmap)

## Overview

**trendy-tube** is a machine learning project that predicts YouTube video success using metadata and channel characteristics. The project processes 85+ million video records from the YouNiverse dataset to identify patterns that distinguish successful videos from others.

**Key Features:**
- End-to-end ML pipeline from raw data to trained models
- Stratified sampling to handle class imbalance
- Feature engineering focused on pre-upload and early signals
- Multiple classification models (Random Forest, Gradient Boosting)
- Designed to prevent data leakage by using only features available at/near upload time

## Folder Structure

```
trendy-tube/
│
├── RawData/                          # Raw datasets (not tracked in git)
│   ├── _raw_yt_metadata.jsonl.zst    # 14.7 GB compressed video metadata (85M+ videos)
│   ├── _raw_df_channels.tsv.gz       # 6.4 MB channel information
│   ├── num_comments.tsv.gz           # 754.6 MB comment counts per video
│   └── yt_metadata_helper.feather    # 2.8 GB helper metadata
│
├── SampleData/                       # Generated sample datasets
│   ├── random_sample_raw_yt_metadata.csv.gz      # Stage 1 output (~850K videos)
│   ├── stratified_sample_raw_yt_metadata.csv.gz  # Stage 2 output (100K videos)
│   ├── data.csv.gz                               # Stage 3 output (engineered features)
│   └── Archive/                                  # Previous sampling experiments
│
├── Preprocessing/                    # Jupyter notebooks for data exploration
│   ├── data_exploration.ipynb        # EDA and initial analysis
│   └── Archive/                      # Old preprocessing notebooks
│
├── Docs/                             # Documentation and visualizations
│   ├── ER_Diagram.png                # Entity-relationship diagram
│   ├── ER_Diagram_Simplified.png     # Simplified ER diagram
│   ├── YouNiverse Large-Scale Channel and Video Metadata.pdf
│   └── target_dist_stratified.png    # Generated: target distribution plot
│
├── models/                           # Saved trained models (generated)
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── scaler.pkl
│
├── pipeline.py                       # Main orchestration script
├── random_sampling.py                # Stage 1: Random sampling from compressed data
├── stratified_sampling.py            # Stage 2: Stratified sampling by engagement
├── feature_engineering.py            # Stage 3: Feature extraction and engineering
├── model_training.py                 # Stage 4: Model training and evaluation
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Python Scripts

### `pipeline.py` - Main Orchestration Script

**Purpose:** Orchestrates the complete end-to-end ML pipeline from raw data to trained models.

**Key Components:**
- `TrendyTubePipeline` class manages the four-stage workflow
- Configurable paths, sampling parameters, and model settings
- Skip flags for iterative development (reuse intermediate outputs)
- Progress tracking and error handling

**Stages:**
1. Random sampling from compressed metadata
2. Stratified sampling based on engagement metrics
3. Feature engineering from video/channel characteristics
4. Model training and evaluation

**Configuration Options:**
```python
config = {
    'random_sampling_ratio': 0.01,      # 1% of 85M videos
    'stratified_sample_size': 100_000,  # Final sample size
    'success_percentile': 90,           # Top 10% = successful
    'test_size': 0.2,                   # 80/20 train/test split
    'random_state': 42,                 # Reproducibility seed
}
```

### `random_sampling.py` - Stage 1: Random Sampling

**Purpose:** Samples from the massive 14.7 GB compressed JSONL file using Bernoulli sampling.

**Key Features:**
- Memory-efficient streaming from Zstandard-compressed files
- Two-pass approach: count rows, then sample
- Early stopping when target sample size is reached
- Handles 85+ million records without loading into memory

**Algorithm:**
- Bernoulli sampling: Each row independently selected with probability `ratio`
- Expected output: ~850K videos (1% of 85M)

**Class:** `RandomSampler`
- `count_rows()`: Count total lines in compressed file
- `sample_rows()`: Perform Bernoulli sampling
- `run_sampling()`: Execute complete sampling workflow

### `stratified_sampling.py` - Stage 2: Stratified Sampling

**Purpose:** Creates a balanced dataset by stratifying on video success (engagement-based).

**Target Definition:**
```
is_successful = 1 if video is in top 10% by engagement_rate
engagement_rate = (like_rate - 0.5 * dislike_rate) / days_since_upload
```

**Key Features:**
- Joins metadata with channel statistics and comment counts
- Calculates time-normalized engagement metrics
- Stratified sampling ensures balanced class distribution (50/50 successful/unsuccessful)
- Generates distribution visualization

**Important Notes:**
- Variable observation windows (different crawl dates)
- Survivor bias (only crawled videos included)
- Not predictive at a fixed time point (e.g., day 7 performance)

**Class:** `StratifiedSampler`
- `calculate_time_normalized_engagement()`: Compute engagement scores
- `create_target_variable()`: Define binary success target
- `stratified_sample()`: Perform stratified split
- `run_sampling_pipeline()`: Execute complete workflow

### `feature_engineering.py` - Stage 3: Feature Engineering

**Purpose:** Extracts predictive features while preventing data leakage.

**Data Leakage Prevention:**
- Only uses features available at upload time or shortly after
- Avoids engagement metrics (views, likes, comments) that contain target information
- Features are pre-upload (channel stats, metadata) or early signals only

**Feature Categories:**

1. **Video Metadata Features:**
   - Title: length, word count, questions, exclamations, uppercase ratio
   - Description: length, word count, presence
   - Tags: count
   - Duration: seconds, categorical bins

2. **Temporal Features:**
   - Upload: day of week, hour, month, year, is weekend
   - Time since channel creation

3. **Channel Features:**
   - Subscriber bins, view bins
   - Channel age
   - Join year/month

4. **Categorical Encodings:**
   - Category ID (one-hot encoding)
   - License type

**Class:** `FeatureEngineer`
- `engineer_features()`: Create all feature columns
- `prepare_features_and_target()`: Split X/y, drop leaky columns
- `run_feature_engineering_pipeline()`: Complete workflow with save

### `model_training.py` - Stage 4: Model Training & Evaluation

**Purpose:** Trains classification models and evaluates their performance.

**Models:**
1. **Random Forest Classifier**
   - 100 trees
   - Max depth: 20
   - Min samples split: 10
   
2. **Gradient Boosting Classifier**
   - 100 estimators
   - Learning rate: 0.1
   - Max depth: 5

**Evaluation Metrics:**
- Classification report (precision, recall, F1-score)
- ROC-AUC score
- Confusion matrix
- Feature importance analysis
- Top 20 most important features

**Outputs:**
- Trained model pickles (`models/`)
- Feature scaler (`models/scaler.pkl`)
- Feature importance plots (`Docs/`)
- Classification reports (console)

**Class:** `ModelTrainer`
- `prepare_train_test_split()`: Split and scale features
- `train_random_forest()`: Train RF classifier
- `train_gradient_boosting()`: Train GB classifier
- `evaluate_model()`: Calculate metrics, generate plots
- `run_training_pipeline()`: Train all models, save outputs

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/engindenizdogu/trendy-tube.git
cd trendy-tube
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Required packages:**
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning models and preprocessing
- `zstandard` - Decompression for .zst files
- `pyarrow` - Feather file format support
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization

3. **Download data:**
   - Place raw datasets in `RawData/` folder
   - Required files: `_raw_yt_metadata.jsonl.zst`, `_raw_df_channels.tsv.gz`, `num_comments.tsv.gz`

## Usage

### Running the Full Pipeline

Execute the complete four-stage workflow:

```bash
python pipeline.py
```

This will:
1. Sample 1% of 85M videos (~850K) from compressed metadata
2. Create a balanced stratified sample of 100K videos
3. Engineer features from metadata and channel info
4. Train Random Forest and Gradient Boosting models
5. Save models and evaluation plots

**Expected Runtime:** ~30-60 minutes (depending on hardware)

### Running Individual Stages

**Stage 1: Random Sampling**
```python
from random_sampling import RandomSampler

sampler = RandomSampler(ratio=0.01, random_state=42)
sample_df = sampler.run_sampling(
    source_path='RawData/_raw_yt_metadata.jsonl.zst',
    output_path='SampleData/random_sample_raw_yt_metadata.csv.gz'
)
```

**Stage 2: Stratified Sampling**
```python
from stratified_sampling import StratifiedSampler

sampler = StratifiedSampler(target_sample_size=100_000, random_state=42)
stratified_df = sampler.run_sampling_pipeline(
    metadata_path='SampleData/random_sample_raw_yt_metadata.csv.gz',
    channels_path='RawData/_raw_df_channels.tsv.gz',
    comments_count_path='RawData/num_comments.tsv.gz',
    output_csv_path='SampleData/stratified_sample_raw_yt_metadata.csv.gz',
    output_plot_path='Docs/target_dist_stratified.png'
)
```

**Stage 3: Feature Engineering**
```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer(random_state=42)
features_df, X, y, feature_names = engineer.run_feature_engineering_pipeline(
    input_path='SampleData/stratified_sample_raw_yt_metadata.csv.gz',
    output_path='SampleData/data.csv.gz'
)
```

**Stage 4: Model Training**
```python
from model_training import ModelTrainer

trainer = ModelTrainer(test_size=0.2, random_state=42)
results = trainer.run_training_pipeline(
    X=X,
    y=y,
    feature_names=feature_names,
    output_dir='models'
)
```

### Skipping Stages (Iterative Development)

To reuse existing intermediate outputs:

```python
from pipeline import TrendyTubePipeline

config = {
    'skip_random_sampling': True,       # Reuse existing random sample
    'skip_stratified_sampling': True,   # Reuse existing stratified sample
    'skip_feature_engineering': False,  # Re-run feature engineering
}

pipeline = TrendyTubePipeline(config=config)
results = pipeline.run_full_pipeline()
```

## Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Random Sampling (random_sampling.py)                   │
│ Input:  _raw_yt_metadata.jsonl.zst (14.7 GB, 85M videos)        │
│ Output: random_sample_raw_yt_metadata.csv.gz (~850K videos)     │
│ Method: Bernoulli sampling with ratio=0.01                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Stratified Sampling (stratified_sampling.py)           │
│ Input:  Random sample + channels + comment counts               │
│ Output: stratified_sample_raw_yt_metadata.csv.gz (100K videos)  │
│ Method: Engagement-based target, stratified split 50/50         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: Feature Engineering (feature_engineering.py)           │
│ Input:  Stratified sample (100K videos)                         │
│ Output: data.csv.gz (100K videos × ~50 features)                │
│ Method: Extract metadata, temporal, channel features            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: Model Training (model_training.py)                     │
│ Input:  Engineered features (X) and target (y)                  │
│ Output: Trained models, evaluation metrics, plots               │
│ Models: Random Forest, Gradient Boosting                        │
└─────────────────────────────────────────────────────────────────┘
```

## Data Assets (repo root / `RawData/`)
| File | Size | Used | Notes |
|------|------|------|-------|
| `_raw_df_channels.tsv.gz` | 6.4 MB | Yes | |
| `_raw_df_timeseries.tsv.gz` | 653.1 MB | No | |
| `_raw_yt_metadata.jsonl.zst` | 14.7 GB | Yes | |
| `df_channels_en.tsv.gz` | 6.0 MB | No | |
| `df_timeseries_en.tsv.gz` | 571.1 MB | No | |
| `num_comments.tsv.gz` | 754.6 MB | Yes | |
| `num_comments_authors.tsv.gz` | 1.4 GB | No | |
| `youtube_comments.tsv.gz` | 77.2 GB | No | File too large |
| `yt_metadata_en.jsonl.gz` | 13.6 GB | No | Same as `_raw_yt_metadata.jsonl.zst` |
| `yt_metadata_helper.feather` | 2.8 GB | No | Contains metadata for `_raw_yt_metadata.jsonl.zst` |

## Project Roadmap

- [x] Create a sample dataset
- [x] Create the preprocessing pipeline
- [] Feature engineering
  - [] One hot encoding
  - [] Scaling
  - [] Feature engineering/extraction/selection
- [x] Model training
  - [x] Random Forest Model
  - [x] Gradient Boosting Model
- [ ] Model Evaluation improvements
  - [ ] Cross-validation
  - [ ] Hyperparameter tuning
  - [ ] Additional models (XGBoost, Neural Networks)
- [ ] Look for additional scraping solutions for more recent data (Optional)
- [ ] Agentic Architecture
- [ ] Frontend

---

## Contributing

This project is part of CS513 Data Mining coursework. For questions or contributions, please contact the repository owner.

## License

This project uses the YouNiverse dataset. Please refer to the original dataset documentation for licensing information.

## Acknowledgments

- **Dataset:** YouNiverse - A Large-Scale Channel and Video Metadata from YouTube
- **Course:** CS513 Data Mining
- **Tools:** scikit-learn, pandas, Python 3.x

