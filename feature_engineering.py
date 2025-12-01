import os
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering for YouTube trend prediction.
    
    IMPORTANT - Data Leakage Prevention:
    - Only uses features available at upload time or shortly after
    - Avoids using engagement metrics (views, likes, comments) that contain target info
    - Features are pre-upload (channel stats, metadata) or early signals only
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
    
    def engineer_features(self, df):
        """
        Create features from video metadata and channel characteristics.
        """
        features_df = df.copy()
        
        # === VIDEO METADATA FEATURES ===
        
        # Title features
        features_df['title_length'] = features_df['title'].fillna('').str.len()
        features_df['title_word_count'] = features_df['title'].fillna('').str.split().str.len()
        features_df['title_has_question'] = features_df['title'].fillna('').str.contains(r'\?').astype(int)
        features_df['title_has_exclamation'] = features_df['title'].fillna('').str.contains('!').astype(int)
        features_df['title_uppercase_ratio'] = features_df['title'].fillna('').apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        
        # Description features
        features_df['description_length'] = features_df['description'].fillna('').str.len()
        features_df['description_word_count'] = features_df['description'].fillna('').str.split().str.len()
        features_df['has_description'] = (features_df['description_length'] > 0).astype(int)
        
        # Tags features (tags are comma-separated strings)
        features_df['num_tags'] = features_df['tags'].fillna('').apply(
            lambda x: len(set(tag.strip().lower() for tag in x.split(',') if tag.strip())) if x else 0
        )
        
        # Video duration (convert to minutes)
        features_df['duration_minutes'] = features_df['duration'].fillna(0) / 60
        features_df['is_short_video'] = (features_df['duration_minutes'] < 5).astype(int)
        features_df['is_long_video'] = (features_df['duration_minutes'] > 20).astype(int)
        
        # Upload timing features
        features_df['upload_day_of_week'] = pd.to_datetime(features_df['upload_date_dt'], format='mixed', errors='coerce').dt.dayofweek
        #features_df['upload_hour'] = features_df['upload_date_dt'].dt.hour
        #features_df['is_weekend_upload'] = (features_df['upload_day_of_week'] >= 5).astype(int)
        
        # === CHANNEL CHARACTERISTICS ===
        # These are pre-existing features (before video upload) - NO DATA LEAKAGE
        
        # Channel size and popularity (known before upload)
        features_df['channel_subscribers'] = features_df['subscribers_cc'].fillna(0)
        features_df['channel_total_videos'] = features_df['videos_cc'].fillna(0)
        #features_df['channel_subscriber_rank'] = features_df['subscriber_rank_sb'].fillna(features_df['subscriber_rank_sb'].max())
        
        # Channel productivity metrics (derived from pre-existing data)
        features_df['subscriber_to_video_ratio'] = features_df['channel_subscribers'] / (
            features_df['channel_total_videos'] + 1
        )
        
        # Log transform for skewed features
        features_df['log_channel_subscribers'] = np.log1p(features_df['channel_subscribers'])
        features_df['log_channel_total_videos'] = np.log1p(features_df['channel_total_videos'])
        
        # Category one-hot encoding
        if 'categories' in features_df.columns:
            # Fill missing values with 'Unknown'
            features_df['categories'] = features_df['categories'].fillna('Unknown')
            
            # One-hot encode categories
            category_dummies = pd.get_dummies(features_df['categories'], prefix='category')
            features_df = pd.concat([features_df, category_dummies], axis=1)
        
        return features_df
    
    def run_feature_engineering_pipeline(self, input_path, output_path):
        """
        Execute complete feature engineering pipeline.
        
        Args:
            input_path: Path to stratified sample CSV.gz
            output_path: Path to save engineered features CSV.gz
            
        Returns:
            DataFrame with engineered features
        """
        print("="*70)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        # Load stratified sample
        print(f"Loading stratified sample from: {input_path}")
        df = pd.read_csv(input_path, compression='gzip', low_memory=False)
        print(f"Loaded {len(df):,} rows")
        
        # Engineer features
        print("\nEngineering features...")
        features_df = self.engineer_features(df)
        
        # Save logging copy with video info before dropping columns
        columns_to_log = ['channel_id', 'title', 'description', 'tags', 'title_length', 'title_word_count', 
                          'title_has_question', 'title_has_exclamation', 'title_uppercase_ratio', 
                          'description_length', 'description_word_count', 'has_description', 'num_tags']
        df_log = features_df[columns_to_log].copy()
        
        # Column cleanup - remove columns not needed for modeling
        columns_to_drop = [
            'upload_date', 'upload_date_dt', 'crawl_date', 'crawl_date_dt', 
            'view_count', 'like_count', 'dislike_count', 'like_rate', 'dislike_rate', 
            'engagement_raw', 'days_since_upload', 'channel_x', 'channel_y', 
            'display_id', 'category_cc', 'join_date', 'name_cc', 'subscriber_rank_sb',
            'title', 'description', 'tags', 'duration', 'engagement_per_day', 
            'channel_id', 'subscribers_cc', 'videos_cc', 'categories'
        ]
        features_df.drop(columns=columns_to_drop, inplace=True)
        
        # Get feature names (excluding target)
        feature_names = [col for col in features_df.columns if col != 'is_successful']
        
        # Save engineered features with target variable and logging file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_log.to_csv(output_path.replace('data', 'data_with_video_info'), index=False, compression='gzip')
        features_df.to_csv(output_path, index=False, compression='gzip')
        print(f"\nEngineered features saved: {output_path}")
        print(f"Total features created: {len(feature_names)}")
        print(f"Features: {feature_names}")
        
        # Print feature summary
        print("\nFeature Summary:")
        print(f"  - Title features: 5")
        print(f"  - Description features: 3")
        print(f"  - Content features: 4")
        print(f"  - Upload timing features: 3")
        print(f"  - Channel features: 6")
        print(f"  - Category features: 1")
        print(f"  Total predictive features: {len(feature_names)}")
        
        return features_df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # Initialize feature engineer
    engineer = FeatureEngineer(random_state=42)
    
    # Run feature engineering pipeline
    features_df = engineer.run_feature_engineering_pipeline(
        input_path='SampleData/stratified_sample_raw_yt_metadata.csv.gz',
        output_path='SampleData/data.csv.gz'
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Output: SampleData/data.csv.gz")
    print(f"Features shape: {features_df.shape}")
