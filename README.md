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
- Random and data preparation sampling with engagement filtering
- Feature engineering with train/test split and scaling
- Multiple classification models with hyperparameter tuning
- Designed to prevent data leakage - labels created only after train/test split

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
│   ├── prepared_data.csv.gz                      # Stage 2 output (100K videos with engagement)
│   ├── X_train.csv.gz, X_test.csv.gz             # Stage 3 output (train/test features)
│   ├── y_train.csv.gz, y_test.csv.gz             # Stage 3 output (train/test labels)
│   ├── scaler.pkl, feature_names.pkl             # Stage 3 output (preprocessing artifacts)
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
├── Models/                           # Saved trained models (generated)
│   ├── best_hyperparameters.txt      # Best hyperparameters for each model
│   ├── evaluation_metrics.csv        # Model performance metrics
│   ├── Plots/                        # Model evaluation plots
│   └── Archive/                      # Previous model runs
│
├── pipeline.py                       # Main orchestration script
├── random_sampling.py                # Stage 1: Random sampling from compressed data
├── data_preparation.py               # Stage 2: Data preparation and engagement calculation
├── feature_engineering.py            # Stage 3: Feature extraction and train/test split
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
2. Data preparation: Merge sources and calculate engagement metrics
3. Feature engineering and train/test split with scaling
4. Model training with label creation and hyperparameter tuning

**Configuration Options:**
```python
config = {
    'random_sampling_ratio': 0.01,          # 1% of 85M videos
    'random_sampling_min_views_per_day': 10,  # Filter low-engagement videos
    'preparation_sample_size': 100_000,     # Downsample if needed
    'success_percentile': 90,               # Top 10% = successful (for labels)
    'test_size': 0.2,                       # 80/20 train/test split
    'random_state': 42,                     # Reproducibility seed
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

### `data_preparation.py` - Stage 2: Data Preparation

**Purpose:** Merges data sources and calculates engagement metrics. Does NOT create labels to prevent data leakage.

**Key Features:**
- Joins metadata with channels, timeseries, and comment counts
- Calculates time-normalized engagement metrics (`engagement_per_day`)
- Optional random downsampling to target size
- Saves prepared data for feature engineering

**Engagement Metric:**
```
engagement_per_day = (
    (avg_rating * view_count) + 
    (like_count - dislike_count) + 
    comment_count
) / days_since_upload
```

**Important Notes:**
- NO label creation at this stage (labels created in model training)
- Preserves all engagement data for later use
- Variable observation windows (different crawl dates)

**Class:** `DataPreparation`
- `calculate_time_normalized_engagement()`: Compute engagement scores
- `merge_data_sources()`: Join all data sources
- `run_preparation_pipeline()`: Execute complete workflow

### `feature_engineering.py` - Stage 3: Feature Engineering & Train/Test Split

**Purpose:** Extracts predictive features, creates train/test split, and applies scaling.

**Key Changes:**
- Creates train/test split BEFORE label creation (in model training)
- Scales features using StandardScaler (fitted on training data only)
- Saves scaled datasets and preprocessing artifacts
- No label creation at this stage

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

**Purpose:** Creates labels from training data only, trains models with hyperparameter tuning.

**Label Creation (Prevents Data Leakage):**
- Labels created AFTER train/test split
- Success threshold calculated from training data's engagement_per_day percentile
- Same threshold applied to test data
- Ensures no information leakage from test set

**Models with GridSearchCV:**
1. **Random Forest Classifier**
2. **Decision Tree Classifier**
3. **Linear SVC**
4. **K-Nearest Neighbors**
5. **Multi-Layer Perceptron**

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
- `get_model_configs()`: Define models and hyperparameter grids
- `train_model_with_tuning()`: Train with GridSearchCV
- `evaluate_model()`: Calculate metrics, generate plots
- `plot_feature_importance()`: Visualize top features
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
1. Sample videos from compressed metadata with view threshold filtering
2. Merge data sources and calculate engagement metrics
3. Engineer features, create train/test split, and apply scaling
4. Create labels from training data and train multiple models with hyperparameter tuning
5. Save models, best hyperparameters, and evaluation plots

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

**Stage 2: Data Preparation**
```python
from data_preparation import DataPreparation

prep = DataPreparation(target_sample_size=100_000, random_state=42)
prepared_df = prep.run_preparation_pipeline(
    metadata_path='SampleData/random_sample_raw_yt_metadata.csv.gz',
    channels_path='RawData/_raw_df_channels.tsv.gz',
    timeseries_path='RawData/_raw_df_timeseries.tsv.gz',
    comments_path='RawData/num_comments.tsv.gz',
    output_csv_path='SampleData/prepared_data.csv.gz'
)
```

**Stage 3: Feature Engineering**
```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer(
    test_size=0.2,
    random_state=42,
    success_percentile=90
)
result = engineer.run_feature_engineering_pipeline(
    input_path='SampleData/prepared_data.csv.gz',
    output_dir='SampleData'
)
# Returns: X_train, X_test, y_train, y_test, feature_names
```

**Stage 4: Model Training**
```python
from model_training import ModelTrainer

trainer = ModelTrainer(random_state=42, n_jobs=-1)
results = trainer.run_training_pipeline(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    feature_names=feature_names,
    output_dir='Models'
)
```

### Skipping Stages (Iterative Development)

To reuse existing intermediate outputs:

```python
from pipeline import TrendyTubePipeline

config = {
    'skip_random_sampling': True,       # Reuse existing random sample
    'skip_data_preparation': True,      # Reuse existing prepared data
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
│ Method: Bernoulli sampling with view threshold filtering        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Data Preparation (data_preparation.py)                 │
│ Input:  Random sample + channels + timeseries + comments        │
│ Output: prepared_data.csv.gz (100K videos with engagement)      │
│ Method: Merge sources, calculate engagement_per_day metric      │
│ Note:   NO LABEL CREATION - prevents data leakage               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: Feature Engineering (feature_engineering.py)           │
│ Input:  Prepared data (100K videos)                             │
│ Output: X_train, X_test, y_train, y_test (scaled)               │
│ Method: Extract features, train/test split, StandardScaler      │
│ Note:   Split BEFORE label creation in next stage               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: Model Training (model_training.py)                     │
│ Input:  Pre-split datasets with engagement_per_day              │
│ Output: Trained models, evaluation metrics, plots               │
│ Method: Create labels from TRAINING data percentile only        │
│ Models: RF, DecisionTree, LinearSVC, KNN, MLP (w/ GridSearch)   │
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
- [x] Feature engineering
  - [x] One hot encoding
  - [x] Scaling
  - [x] Feature engineering/extraction/selection
- [x] Model training
  - [x] Random Forest Model
  - [x] Gradient Boosting Model
- [ ] Model Evaluation improvements
  - [ ] Cross-validation
  - [ ] Hyperparameter tuning
  - [x] Additional models (XGBoost, Neural Networks)
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

