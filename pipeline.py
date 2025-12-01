"""
YouTube Trendy Tube - Complete ML Pipeline
==========================================

This pipeline orchestrates the complete workflow for predicting YouTube video success:
1. Random sampling from raw compressed metadata
2. Stratified sampling based on engagement metrics
3. Feature engineering from video and channel characteristics
4. Model training and evaluation
"""

import os
import time
import pandas as pd
from datetime import datetime

# Import custom modules
from random_sampling import RandomSampler
from stratified_sampling import StratifiedSampler
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer


class TrendyTubePipeline:
    """
    Complete end-to-end pipeline for YouTube trend prediction.
    
    Pipeline stages:
    1. Random Sampling: Sample from 85M+ videos using Bernoulli sampling
    2. Stratified Sampling: Create balanced dataset based on engagement
    3. Feature Engineering: Extract and engineer predictive features
    4. Model Training: Train and evaluate classification models
    """
    
    def __init__(self, config=None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Dictionary with pipeline configuration parameters
        """
        self.config = config or self._default_config()
        self.start_time = None
        self.results = {}
        
    def _default_config(self):
        """
        Default pipeline configuration.
        """
        return {
            # Data paths
            'raw_metadata_path': 'RawData/_raw_yt_metadata.jsonl.zst',
            'channels_path': 'RawData/_raw_df_channels.tsv.gz',
            'timeseries_path': 'RawData/_raw_df_timeseries.tsv.gz',

            
            # Output paths
            'random_sample_output': 'SampleData/random_sample_raw_yt_metadata.csv.gz',
            'stratified_sample_output': 'SampleData/stratified_sample_raw_yt_metadata.csv.gz',
            'engineered_features_output': 'SampleData/data.csv.gz',
            'models_output_dir': 'models',
            'plots_output_dir': 'Docs',
            
            # Sampling parameters
            'random_sampling_ratio': 0.01,  # 1% of 85M = ~850K videos
            'random_sampling_max_lines': None,
            'stratified_sample_size': 100_000,
            'success_percentile': 90,  # Top 10% considered successful
            
            # Model parameters
            'test_size': 0.2,
            'random_state': 42,
            
            # Pipeline control
            'skip_random_sampling': False,  # Set True if random sample exists
            'skip_stratified_sampling': False,  # Set True if stratified sample exists
            'skip_feature_engineering': False,  # Set True if features exist
        }
    
    def print_stage_header(self, stage_name, stage_number, total_stages=4):
        """
        Print formatted stage header.
        """
        print("\n" + "="*80)
        print(f"STAGE {stage_number}/{total_stages}: {stage_name.upper()}")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def stage1_random_sampling(self):
        """
        Stage 1: Random sampling from compressed metadata.
        """
        self.print_stage_header("Random Sampling", 1)
        
        if self.config['skip_random_sampling'] and os.path.exists(self.config['random_sample_output']):
            print(f"Skipping random sampling - using existing file:")
            print(f"  {self.config['random_sample_output']}")
            random_df = pd.read_csv(self.config['random_sample_output'], compression='gzip')
            print(f"Loaded {len(random_df):,} rows from existing random sample")
            self.results['random_sample'] = random_df
            return random_df
        
        # Initialize sampler
        sampler = RandomSampler(
            ratio=self.config['random_sampling_ratio'],
            max_lines=self.config['random_sampling_max_lines'],
            random_state=self.config['random_state']
        )
        
        # Run sampling
        random_df = sampler.run_sampling(
            source_path=self.config['raw_metadata_path'],
            output_path=self.config['random_sample_output']
        )
        
        self.results['random_sample'] = random_df
        print(f"\n✓ Stage 1 completed: {len(random_df):,} videos sampled")
        return random_df
    
    def stage2_stratified_sampling(self):
        """
        Stage 2: Stratified sampling based on engagement metrics.
        """
        self.print_stage_header("Stratified Sampling", 2)
        
        if self.config['skip_stratified_sampling'] and os.path.exists(self.config['stratified_sample_output']):
            print(f"Skipping stratified sampling - using existing file:")
            print(f"  {self.config['stratified_sample_output']}")
            stratified_df = pd.read_csv(self.config['stratified_sample_output'], compression='gzip')
            print(f"Loaded {len(stratified_df):,} rows from existing stratified sample")
            self.results['stratified_sample'] = stratified_df
            return stratified_df
        
        # Initialize stratified sampler
        sampler = StratifiedSampler(
            target_sample_size=self.config['stratified_sample_size'],
            success_percentile=self.config['success_percentile'],
            random_state=self.config['random_state']
        )
        
        # Run stratified sampling pipeline
        stratified_df = sampler.run_sampling_pipeline(
            metadata_path=self.config['random_sample_output'],
            channels_path=self.config['channels_path'],
            timeseries_path=self.config['timeseries_path'],
            output_csv_path=self.config['stratified_sample_output'],
            output_plot_path=os.path.join(
                self.config['plots_output_dir'], 
                'target_dist_stratified.png'
            )
        )
        
        self.results['stratified_sample'] = stratified_df
        print(f"\n✓ Stage 2 completed: {len(stratified_df):,} videos in stratified sample")
        return stratified_df
    
    def stage3_feature_engineering(self):
        """
        Stage 3: Feature engineering from video and channel metadata.
        """
        self.print_stage_header("Feature Engineering", 3)
        
        if self.config['skip_feature_engineering'] and os.path.exists(self.config['engineered_features_output']):
            print(f"Skipping feature engineering - using existing file:")
            print(f"  {self.config['engineered_features_output']}")
            features_df = pd.read_csv(self.config['engineered_features_output'], compression='gzip')
            print(f"Loaded {len(features_df):,} rows with engineered features")
            
            self.results['features_df'] = features_df
            return features_df
        
        # Load stratified sample
        stratified_df = self.results.get('stratified_sample')
        if stratified_df is None:
            stratified_df = pd.read_csv(self.config['stratified_sample_output'], compression='gzip')
        
        # Initialize feature engineer
        engineer = FeatureEngineer(random_state=self.config['random_state'])
        
        # Run feature engineering pipeline
        print("Engineering features from video and channel metadata...")
        features_df = engineer.run_feature_engineering_pipeline(
            input_path=self.config['stratified_sample_output'],
            output_path=self.config['engineered_features_output']
        )
        
        self.results['features_df'] = features_df
        
        print(f"\n✓ Stage 3 completed: {features_df.shape[1] - 1} features engineered")
        return features_df
    
    def stage4_model_training(self):
        """
        Stage 4: Train and evaluate classification models.
        """
        self.print_stage_header("Model Training & Evaluation", 4)
        
        # Get features dataframe
        features_df = self.results.get('features_df')
        
        if features_df is None:
            # Load from file if not in results
            features_df = pd.read_csv(self.config['engineered_features_output'], compression='gzip')
        
        # Prepare features and target
        print("Preparing features and target variable...")
        y = features_df['is_successful']
        X = features_df.drop(columns=['is_successful'])
        feature_names = list(X.columns)
        
        # Handle missing and infinite values
        import numpy as np
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Features used: {len(feature_names)} features")
        
        # Initialize model trainer
        trainer = ModelTrainer(
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # Run training pipeline
        training_results = trainer.run_training_pipeline(
            X=X,
            y=y,
            feature_names=feature_names,
            output_dir=self.config['models_output_dir']
        )
        
        self.results['training_results'] = training_results
        
        print(f"\n✓ Stage 4 completed: Models trained and saved to {self.config['models_output_dir']}")
        return training_results
    
    def run_full_pipeline(self):
        """
        Execute complete end-to-end pipeline.
        """
        self.start_time = time.time()
        
        print("\n" + "="*80)
        print("TRENDY TUBE - YOUTUBE TREND PREDICTION PIPELINE")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration:")
        for key, value in self.config.items():
            if not key.endswith('_path') and not key.startswith('skip_'):
                print(f"  {key}: {value}")
        
        try:
            # Stage 1: Random Sampling
            self.stage1_random_sampling()
            
            # Stage 2: Stratified Sampling
            self.stage2_stratified_sampling()
            
            # Stage 3: Feature Engineering
            self.stage3_feature_engineering()
            
            # Stage 4: Model Training
            self.stage4_model_training()
            
            # Pipeline completion
            duration = time.time() - self.start_time
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Total elapsed time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print(f"\nOutputs:")
            print(f"  Random sample: {self.config['random_sample_output']}")
            print(f"  Stratified sample: {self.config['stratified_sample_output']}")
            print(f"  Engineered features: {self.config['engineered_features_output']}")
            print(f"  Models: {self.config['models_output_dir']}/")
            print(f"  Plots: {self.config['plots_output_dir']}/")
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed at stage with error:")
            print(f"  {type(e).__name__}: {str(e)}")
            raise


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Initialize and run pipeline with default configuration
    # To skip stages, modify the _default_config() method in the class

    config = {
        # Data paths
        'raw_metadata_path': 'RawData/_raw_yt_metadata.jsonl.zst',
        'channels_path': 'RawData/_raw_df_channels.tsv.gz',
        'timeseries_path': 'RawData/_raw_df_timeseries.tsv.gz',
            
        # Output paths
        'random_sample_output': 'SampleData/random_sample_raw_yt_metadata.csv.gz',
        'stratified_sample_output': 'SampleData/stratified_sample_raw_yt_metadata.csv.gz',
        'engineered_features_output': 'SampleData/data.csv.gz',
        'models_output_dir': 'models',
        'plots_output_dir': 'Docs',
            
        # Sampling parameters
        'random_sampling_ratio': 0.01,  # 1% of 85M = ~850K videos
        'random_sampling_max_lines': None,
        'stratified_sample_size': 100_000,
        'success_percentile': 90,  # Top 10% considered successful
            
        # Model parameters
        'test_size': 0.2,
        'random_state': 42,
            
        # Pipeline control
        'skip_random_sampling': True,  # Set True if random sample exists
        'skip_stratified_sampling': False,  # Set True if stratified sample exists
        'skip_feature_engineering': False,  # Set True if features exist
    }

    pipeline = TrendyTubePipeline(config=config)
    results = pipeline.run_full_pipeline()
    #print(results)