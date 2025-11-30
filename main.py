import time
import pandas as pd
import numpy as np
import json
import gzip
import zstandard as zstd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class YouTubeTrendPredictor:
    """
    ML Pipeline to predict if a YouTube video will be successful (trending)
    based on engagement metrics at 7 days post-upload.
    """
    
    def __init__(self, target_sample_size=100000, success_percentile=90, 
                 chunk_size=10000, random_state=42):
        self.target_sample_size = target_sample_size
        self.success_percentile = success_percentile
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_engagement_score(self, row):
        """
        Create composite engagement score from likes, views, and comments.
        Normalize by views to get engagement rate.
        """
        views = row.get('view_count', 1)
        if views == 0:
            views = 1
        
        likes = row.get('like_count', 0)
        dislikes = row.get('dislike_count', 0)
        
        # Engagement rate components
        like_rate = likes / views
        dislike_rate = dislikes / views
        
        # Weighted engagement score (likes positive, dislikes negative)
        engagement_score = like_rate - (0.5 * dislike_rate)
        
        return engagement_score
    
    def filter_videos_at_7_days(self, metadata_chunk, timeseries_df):
        """
        Filter videos that have at least 7 days of data and calculate
        metrics at exactly the 7-day mark.
        """
        filtered_videos = []
        
        for _, video in metadata_chunk.iterrows():
            video_id = video['display_id']
            upload_date = pd.to_datetime(video['upload_date'])
            
            # Get timeseries for this video
            video_ts = timeseries_df[timeseries_df['channel'] == video_id].copy()
            
            if len(video_ts) == 0:
                continue
            
            video_ts['datetime'] = pd.to_datetime(video_ts['datetime'])
            video_ts['days_since_upload'] = (video_ts['datetime'] - upload_date).dt.total_seconds() / 86400
            
            # Find metrics closest to 7 days
            seven_day_data = video_ts[
                (video_ts['days_since_upload'] >= 6.5) & 
                (video_ts['days_since_upload'] <= 7.5)
            ]
            
            if len(seven_day_data) > 0:
                # Take the closest point to 7 days
                closest_idx = (seven_day_data['days_since_upload'] - 7).abs().idxmin()
                metrics_at_7d = seven_day_data.loc[closest_idx]
                
                video_dict = video.to_dict()
                video_dict['views_at_7d'] = metrics_at_7d['views']
                video_dict['likes_at_7d'] = metrics_at_7d.get('likes', 0)
                video_dict['engagement_at_7d'] = self.create_engagement_score({
                    'view_count': metrics_at_7d['views'],
                    'like_count': metrics_at_7d.get('likes', 0),
                    'dislike_count': 0
                })
                
                filtered_videos.append(video_dict)
        
        return pd.DataFrame(filtered_videos)
    
    def load_zst_file_streaming(self, filepath, buffer_size=65536):
        """
        Load .zst (Zstandard) compressed file with true streaming.
        Yields lines one at a time without loading entire file into memory.
        """
        dctx = zstd.ZstdDecompressor()
        
        with open(filepath, 'rb') as compressed:
            with dctx.stream_reader(compressed) as reader:
                buffer = b''
                
                while True:
                    chunk = reader.read(buffer_size)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    
                    # Process complete lines
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        if line_str:
                            yield line_str
                
                # Process remaining buffer
                if buffer:
                    line_str = buffer.decode('utf-8', errors='ignore').strip()
                    if line_str:
                        yield line_str
    
    def load_data_with_sampling(self, metadata_path, channels_path, 
                                 timeseries_path, comments_count_path):
        """
        Load data in chunks and perform stratified sampling to get 100K videos.
        Stratify by category to maintain distribution.
        """
        print("Loading channels data...")
        channels_df = pd.read_csv(channels_path, sep='\t', compression='gzip')
        
        print("Loading timeseries data in chunks...")
        timeseries_chunks = []
        for chunk in pd.read_csv(timeseries_path, sep='\t', compression='gzip', 
                                  chunksize=self.chunk_size):
            timeseries_chunks.append(chunk)
        timeseries_df = pd.concat(timeseries_chunks, ignore_index=True)
        
        print("Loading comments count data...")
        comments_df = pd.read_csv(comments_count_path, sep='\t', compression='gzip')
        
        print("Processing metadata in chunks...")
        metadata_samples = []
        category_counts = defaultdict(int)
        
        # First pass: get category distribution (streaming, no memory buildup)
        print("First pass: analyzing category distribution...")
        total_videos = 0
        
        for line in self.load_zst_file_streaming(metadata_path):
            try:
                data = json.loads(line)
                category = data.get('categories', ['Unknown'])[0] if data.get('categories') else 'Unknown'
                category_counts[category] += 1
                total_videos += 1
                
                if total_videos % 100000 == 0:
                    print(f"Analyzed {total_videos} videos...")
            except json.JSONDecodeError:
                continue
        
        # Calculate sampling probability for each category (stratified)
        sampling_probs = {}
        for cat, count in category_counts.items():
            sampling_probs[cat] = min(1.0, (self.target_sample_size * count / total_videos) / count)
        
        print(f"\nTotal videos: {total_videos}")
        print(f"Category distribution: {dict(category_counts)}")
        print(f"Sampling probabilities: {sampling_probs}")
        
        # Second pass: sample videos (streaming, one line at a time)
        print("\nSecond pass: sampling videos...")
        sampled_videos = []
        video_count = 0
        
        for line in self.load_zst_file_streaming(metadata_path):
            try:
                data = json.loads(line)
                category = data.get('categories', ['Unknown'])[0] if data.get('categories') else 'Unknown'
                
                # Stratified sampling
                if np.random.random() < sampling_probs[category]:
                    sampled_videos.append(data)
                
                video_count += 1
                if video_count % 100000 == 0:
                    print(f"Processed {video_count} videos, sampled {len(sampled_videos)}...")
                
                # Stop early if we have enough samples
                if len(sampled_videos) >= self.target_sample_size * 1.5:
                    break
            except json.JSONDecodeError:
                continue
        
        metadata_df = pd.DataFrame(sampled_videos)
        print(f"Sampled {len(metadata_df)} videos")
        
        # Clear the sampled_videos list to free memory
        sampled_videos = None
        
        # Filter for videos with 7-day data
        print("\nFiltering videos with 7-day metrics...")
        filtered_df = self.filter_videos_at_7_days(metadata_df, timeseries_df)
        
        # Clear metadata_df to free memory
        metadata_df = None
        
        # Merge with channels and comments
        print("Merging with channel and comment data...")
        merged_df = filtered_df.merge(
            channels_df, 
            left_on='channel_id', 
            right_on='channel',
            how='left',
            suffixes=('', '_channel')
        )
        
        merged_df = merged_df.merge(
            comments_df,
            left_on='display_id',
            right_on='display_id',
            how='left'
        )
        
        return merged_df
    
    def create_target_variable(self, df):
        """
        Create binary target: 1 if video is in top 10% by engagement, 0 otherwise.
        """
        engagement_threshold = df['engagement_at_7d'].quantile(self.success_percentile / 100)
        df['is_successful'] = (df['engagement_at_7d'] >= engagement_threshold).astype(int)
        
        print(f"\nTarget variable distribution:")
        print(df['is_successful'].value_counts())
        print(f"\nEngagement threshold (top {100-self.success_percentile}%): {engagement_threshold:.6f}")
        
        return df
    
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
        
        # Tags features
        features_df['num_tags'] = features_df['tags'].fillna('').apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # Video duration (convert to minutes)
        features_df['duration_minutes'] = features_df['duration'].fillna(0) / 60
        features_df['is_short_video'] = (features_df['duration_minutes'] < 5).astype(int)
        features_df['is_long_video'] = (features_df['duration_minutes'] > 20).astype(int)
        
        # Upload timing features
        features_df['upload_date_dt'] = pd.to_datetime(features_df['upload_date'])
        features_df['upload_day_of_week'] = features_df['upload_date_dt'].dt.dayofweek
        features_df['upload_hour'] = features_df['upload_date_dt'].dt.hour
        features_df['is_weekend_upload'] = (features_df['upload_day_of_week'] >= 5).astype(int)
        
        # === CHANNEL CHARACTERISTICS ===
        
        # Channel size and popularity
        features_df['channel_subscribers'] = features_df['subscribers_cc'].fillna(0)
        features_df['channel_total_videos'] = features_df['videos_cc'].fillna(0)
        features_df['channel_subscriber_rank'] = features_df['subscriber_rank_sb'].fillna(features_df['subscriber_rank_sb'].max())
        
        # Channel engagement metrics
        features_df['avg_views_per_video'] = features_df['channel_subscribers'] / (features_df['channel_total_videos'] + 1)
        features_df['subscriber_to_video_ratio'] = features_df['channel_subscribers'] / (features_df['channel_total_videos'] + 1)
        
        # === EARLY ENGAGEMENT INDICATORS ===
        features_df['num_comments_7d'] = features_df['num_comms'].fillna(0)
        features_df['comments_per_view'] = features_df['num_comments_7d'] / (features_df['views_at_7d'] + 1)
        
        # Category encoding
        if 'categories' in features_df.columns:
            features_df['category'] = features_df['categories'].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown'
            )
            
            if 'category' not in self.label_encoders:
                self.label_encoders['category'] = LabelEncoder()
                features_df['category_encoded'] = self.label_encoders['category'].fit_transform(
                    features_df['category'].fillna('Unknown')
                )
            else:
                features_df['category_encoded'] = self.label_encoders['category'].transform(
                    features_df['category'].fillna('Unknown')
                )
        
        return features_df
    
    def prepare_features_and_target(self, df):
        """
        Select final features and prepare for modeling.
        """
        feature_columns = [
            # Title features
            'title_length', 'title_word_count', 'title_has_question', 
            'title_has_exclamation', 'title_uppercase_ratio',
            
            # Description features
            'description_length', 'description_word_count', 'has_description',
            
            # Tags and content
            'num_tags', 'duration_minutes', 'is_short_video', 'is_long_video',
            
            # Upload timing
            'upload_day_of_week', 'upload_hour', 'is_weekend_upload',
            
            # Channel characteristics
            'channel_subscribers', 'channel_total_videos', 'channel_subscriber_rank',
            'avg_views_per_video', 'subscriber_to_video_ratio',
            
            # Early engagement
            'num_comments_7d', 'comments_per_view',
            
            # Category
            'category_encoded'
        ]
        
        # Filter to only existing columns
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns].fillna(0)
        y = df['is_successful']
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Features used: {feature_columns}")
        
        return X, y, feature_columns
    
    def train_model(self, X, y):
        """
        Train Random Forest and Gradient Boosting classifiers.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n" + "="*50)
        print("Training Random Forest Classifier...")
        print("="*50)
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_rf = rf_model.predict(X_test_scaled)
        y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nRandom Forest Results:")
        print(classification_report(y_test, y_pred_rf))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
        
        print("\n" + "="*50)
        print("Training Gradient Boosting Classifier...")
        print("="*50)
        
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        
        gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_gb = gb_model.predict(X_test_scaled)
        y_pred_proba_gb = gb_model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nGradient Boosting Results:")
        print(classification_report(y_test, y_pred_gb))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_gb):.4f}")
        
        return rf_model, gb_model, X_test_scaled, y_test
    
    def plot_feature_importance(self, model, feature_names, model_name="Model"):
        """
        Plot feature importance.
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def run_pipeline(self, metadata_path, channels_path, timeseries_path, comments_count_path):
        """
        Execute the full ML pipeline.
        """
        print("="*70)
        print("YOUTUBE VIDEO SUCCESS PREDICTION PIPELINE")
        print("="*70)
        
        # Load and sample data
        df = self.load_data_with_sampling(
            metadata_path, channels_path, timeseries_path, comments_count_path
        )
        
        # Create target variable
        df = self.create_target_variable(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Prepare features and target
        X, y, feature_names = self.prepare_features_and_target(df)
        
        # Train models
        rf_model, gb_model, X_test, y_test = self.train_model(X, y)
        
        # Plot feature importances
        print("\nGenerating feature importance plots...")
        self.plot_feature_importance(rf_model, feature_names, "Random Forest")
        self.plot_feature_importance(gb_model, feature_names, "Gradient Boosting")
        
        return rf_model, gb_model, df

# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    start_time = time.time()
    # Initialize predictor
    predictor = YouTubeTrendPredictor(
        target_sample_size=5000,
        success_percentile=90,
        chunk_size=10000,
        random_state=42
    )
    
    # Run pipeline (Total lines: 85,421,645)
    rf_model, gb_model, processed_df = predictor.run_pipeline(
        metadata_path='RawData/_raw_yt_metadata.jsonl.zst',
        channels_path='RawData/_raw_df_channels.tsv.gz',
        timeseries_path='RawData/_raw_df_timeseries.tsv.gz',
        comments_count_path='RawData/num_comments.tsv.gz'
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nPipeline execution time: {elapsed_time:.2f} seconds")
    print(f"Models trained on {len(processed_df)} videos")
    print(f"Target: Top {100 - predictor.success_percentile}% by engagement at 7 days")