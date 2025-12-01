import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class StratifiedSampler:
    """
    Load data, create engagement-based target variable, and perform stratified sampling.
    
    TARGET DEFINITION:
    'is_successful' = 1 if video is in top 10% by engagement rate per day
    
    Engagement rate = (like_rate - 0.5 * dislike_rate) / days_since_upload
    
    INTERPRETATION:
    This captures videos with strong early engagement momentum, normalized 
    by video age. Suitable for identifying videos that gain traction quickly.
    
    LIMITATIONS:
    - Variable observation windows (crawl dates differ)
    - Not predictive at a fixed time point (e.g., day 7)
    - Survivor bias (only crawled videos included)
    
    Outputs a stratified sample CSV.gz file ready for feature engineering.
    """
    
    def __init__(self, target_sample_size=100000, success_percentile=90, random_state=42):
        self.target_sample_size = target_sample_size
        self.success_percentile = success_percentile
        self.random_state = random_state
    
    def calculate_time_normalized_engagement(self, df):
        """
        Calculate engagement score normalized by video age.
        
        Uses crawl_date - upload_date to normalize engagement metrics.
        This accounts for how long the video has been live, making newer 
        and older videos comparable.
        
        Engagement components:
        - Like rate: likes / views
        - Dislike rate: dislikes / views  
        - Weighted score: like_rate - 0.5 * dislike_rate
        - Time normalization: divide by days_since_upload to get rate per day
        """
        print("Calculating time-normalized engagement scores...")
        
        # Convert dates (use mixed format to handle variations)
        df['upload_date_dt'] = pd.to_datetime(df['upload_date'], format='mixed', errors='coerce')
        df['crawl_date_dt'] = pd.to_datetime(df['crawl_date'], format='mixed', errors='coerce')
        
        # Remove rows with invalid dates
        initial_count = len(df)
        df = df.dropna(subset=['upload_date_dt', 'crawl_date_dt']).copy()
        if len(df) < initial_count:
            print(f"Removed {initial_count - len(df):,} videos with invalid dates")
        
        # Calculate days since upload (video age at crawl time)
        df['days_since_upload'] = (df['crawl_date_dt'] - df['upload_date_dt']).dt.total_seconds() / 86400
        
        # Filter out invalid videos (crawled before upload or same-day uploads)
        df = df[df['days_since_upload'] >= 1.0].copy()
        print(f"Filtered to {len(df):,} videos with valid upload/crawl dates (>= 1 day)")
        
        # Handle missing values and zeros
        df['view_count'] = df['view_count'].fillna(1).replace(0, 1)
        df['like_count'] = df['like_count'].fillna(0)
        df['dislike_count'] = df['dislike_count'].fillna(0)
        
        # Calculate engagement rates
        df['like_rate'] = df['like_count'] / df['view_count']
        df['dislike_rate'] = df['dislike_count'] / df['view_count']
        
        # Weighted engagement score (likes positive, dislikes negative)
        df['engagement_raw'] = df['like_rate'] - (0.5 * df['dislike_rate'])
        
        # Normalize by video age to get engagement rate per day
        df['engagement_per_day'] = df['engagement_raw'] / df['days_since_upload']
        
        #print(f"Engagement score stats:")
        #print(df['engagement_per_day'].describe())
        
        return df
    
    def load_data_with_sampling(self, metadata_path, channels_path, timeseries_path):
        """
        Load precomputed random sample and merge with channels and timeseries data.
        Engagement is calculated directly from metadata.
        """
        print("Loading channels data...")
        channels_df = pd.read_csv(channels_path, sep='\t', compression='gzip')

        print("Loading timeseries data...")
        timeseries_df = pd.read_csv(timeseries_path, sep='\t', compression='gzip')

        # Load precomputed random sample CSV.gz for metadata
        print("Loading precomputed random sample metadata...")
        metadata_df = pd.read_csv(metadata_path, compression='gzip', low_memory=False)
        print(f"Loaded metadata sample: {len(metadata_df):,} rows")

        # Calculate time-normalized engagement directly from metadata
        metadata_df = self.calculate_time_normalized_engagement(metadata_df)

        # Merge with channels and comments
        print("\nMerging with channel data...")
        merged_df = metadata_df.merge(
            channels_df,
            left_on='channel_id',
            right_on='channel',
            how='left',
            suffixes=('', '_channel')
        )

        # Group by channel and get the earliest and latest records
        timeseries_df.drop(columns=['category','delta_views','delta_subs','delta_videos','activity','subs','videos'], axis=1, inplace=True)
        timeseries_df['views'] = timeseries_df['views'].astype(int)
        timeseries_df['datetime'] = pd.to_datetime(timeseries_df['datetime'])
        groupby_channels = timeseries_df.groupby('channel')
        latest = timeseries_df.loc[groupby_channels['datetime'].idxmax()].set_index('channel')

        # Build summary with aligned indexes to avoid length/index mismatch
        channel_views = pd.DataFrame({
            'channel': latest.index,
            'channel_views': latest['views']
        }).reset_index(drop=True)

        merged_df = merged_df.merge(
            channel_views,
            left_on=['channel_id'],
            right_on=['channel'],
            how='left'
        )
        
        print(f"Final merged dataset: {len(merged_df):,} rows")
        return merged_df

    def stratified_downsample_by_target(self, df: pd.DataFrame, target_col: str = 'is_successful', 
                                         size: int = 100000) -> pd.DataFrame:
        """
        Create a stratified sample of `size` rows based on the binary target variable.
        Preserves class balance between 0/1 in `target_col`.
        """

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found for stratified sampling")

        if size >= len(df):
            return df.copy()

        sss = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=self.random_state)
        X = np.zeros((len(df), 1))
        y = df[target_col].values
        for _, test_idx in sss.split(X, y):
            return df.iloc[test_idx].copy()
        return df.sample(n=size, random_state=self.random_state)
    
    def create_target_variable(self, df):
        """
        Create binary target: 1 if video is in top 10% by time-normalized engagement, 0 otherwise.
        """
        engagement_threshold = df['engagement_per_day'].quantile(self.success_percentile / 100)
        df['is_successful'] = (df['engagement_per_day'] >= engagement_threshold).astype(int)
        
        print(f"\nTarget variable distribution:")
        print(df['is_successful'].value_counts())
        print(f"\nEngagement threshold (top {100-self.success_percentile}%): {engagement_threshold:.6e}")
        
        return df
    
    def plot_target_distribution(self, df, output_path='target_distribution.png'):
        """Plot and save target variable distribution."""
        plt.figure(figsize=(10, 6))
        counts = df['is_successful'].value_counts().sort_index()
        
        # Bar plot
        ax = counts.plot(kind='bar', color=['#3498db', '#e74c3c'])
        plt.title(f'Target Variable Distribution\n(Top {100-self.success_percentile}% by Engagement)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Is Successful', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks([0, 1], ['Not Successful (0)', 'Successful (1)'], rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(counts):
            ax.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Add percentage
        total = len(df)
        for i, v in enumerate(counts):
            pct = (v / total) * 100
            ax.text(i, v/2, f'{pct:.1f}%', ha='center', va='center', 
                   fontsize=11, color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Target distribution plot saved: {output_path}")
        plt.close()
    
    def run_sampling_pipeline(self, metadata_path, channels_path, timeseries_path, output_csv_path, output_plot_path):
        """Execute data loading, target creation, stratified sampling, and save outputs."""
        print("="*70)
        print("STRATIFIED SAMPLING PIPELINE")
        print("="*70)
        
        # Load and merge data (no timeseries needed)
        df = self.load_data_with_sampling(metadata_path, channels_path, timeseries_path)
        
        # Create target variable
        df = self.create_target_variable(df)
        
        # Plot distribution before sampling
        print("\nPlotting target distribution before sampling...")
        temp_plot = output_plot_path.replace('.png', '_before_sampling.png')
        self.plot_target_distribution(df, temp_plot)
        
        # Stratified downsampling by target to requested size
        if self.target_sample_size is not None and self.target_sample_size < len(df):
            print(f"\nStratified downsampling to {self.target_sample_size:,} by target variable...")
            df = self.stratified_downsample_by_target(
                df, target_col='is_successful', size=self.target_sample_size
            )
            print(f"Downsampled dataset size: {len(df):,}")
            
            # Print final distribution
            print(f"\nFinal target variable distribution:")
            print(df['is_successful'].value_counts())
        
        # Plot final distribution
        print("\nPlotting final target distribution...")
        self.plot_target_distribution(df, output_plot_path)
        
        # Save stratified sample
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False, compression='gzip')
        print(f"\nStratified sample saved: {output_csv_path}")
        print(f"Sample size: {len(df):,} rows")
        
        return df

# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    # Initialize sampler
    sampler = StratifiedSampler(
        target_sample_size=100000,
        success_percentile=90,
        random_state=42
    )
    
    # Run sampling pipeline
    stratified_df = sampler.run_sampling_pipeline(
        metadata_path='SampleData/random_sample_raw_yt_metadata.csv.gz',
        channels_path='RawData/_raw_df_channels.tsv.gz',
        timeseries_path='RawData/_raw_df_timeseries.tsv.gz',
        output_csv_path='SampleData/stratified_sample_raw_yt_metadata.csv.gz',
        output_plot_path='Docs/target_dist_stratified.png'
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n" + "="*70)
    print("SAMPLING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nElapsed time: {elapsed_time:.2f} seconds")
    print(f"Final sample size: {len(stratified_df):,} videos")
    print(f"Target: Top {100 - sampler.success_percentile}% by engagement at 7 days")